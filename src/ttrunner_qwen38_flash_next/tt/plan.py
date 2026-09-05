"""Device residency plan for the 4 x Blackhole p150a mesh.

Two constraints drive the whole design:

* 128 GB of device DRAM total (4 x 32 GB), against a 93.7 GB checkpoint;
* Blackhole's matmul consumes ``bfloat8_b`` / ``bfloat4_b`` block-float natively,
  so the GGUF's IQ3_S / IQ4_NL codebook formats have to be re-encoded anyway.

The n-gram table is the key observation: it is 28.8 GB of the checkpoint but is
a pure gather touching 16 rows per token, so it stays in host RAM and only the
gathered rows cross PCIe. That removes nearly a third of the checkpoint from the
device budget before any quantisation choice is made.

Per-tensor precision follows the same sensitivity ordering Unsloth's
imatrix-calibrated UD-IQ4_XS uses (measured in iteration 002), which is why the
device policy is derived from the source quant rather than invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Residency(Enum):
    DEVICE = "device"  # sharded or replicated across the mesh
    HOST = "host"  # gathered on the host, rows pushed per token


class Shard(Enum):
    REPLICATE = "replicate"  # every device holds a full copy
    COLUMN = "column"  # split output features (dim 0 of an (out, in) weight)
    ROW = "row"  # split input features (dim 1) -- output needs all-reduce
    EXPERT_COLUMN = "expert_column"  # (E, out, in): split `out`
    EXPERT_ROW = "expert_row"  # (E, out, in): split `in`, all-reduce after
    # (E, out, in): split `E`. Every device gets whole experts instead of a slice
    # of all of them, which is the difference between `sparse_matmul` writing
    # [1, 512, M, K] a layer and [1, 128, M, K] -- 84 MB against 21. Same bytes of
    # weight per device either way; a quarter of the output nobody reads. Each
    # device then holds a partial sum over its own experts, so the MoE's existing
    # all-reduce is what makes it whole.
    EXPERT = "expert"
    # The 10240-wide DeltaNet channel axis is [q(2048) | k(2048) | v(6144)];
    # a flat 4-way split at 2560 cuts through the q/k boundary and mispairs
    # heads, so each part is split by head and re-concatenated per device.
    HEAD_QKV_COLUMN = "head_qkv_column"  # channels on the last dim (attn_qkv)
    HEAD_QKV_ROW = "head_qkv_row"  # channels on dim -2 (ssm_conv1d)


@dataclass(frozen=True, slots=True)
class TensorPlan:
    pattern: str
    residency: Residency
    dtype: str  # ttnn dtype name
    shard: Shard
    note: str = ""


# Ordered; first match wins.
PLAN: tuple[TensorPlan, ...] = (
    # -- the 51 B-parameter n-gram table: a gather, never a matmul ------------
    TensorPlan(
        r"^per_layer_token_embd\.weight$", Residency.HOST, "source", Shard.REPLICATE,
        "28.8 GB; 16 rows touched per token, so it never needs to be resident. "
        "Stays IQ4_NL in the mmap and is dequantised per gathered row -- "
        "materialising it as f32 would cost 205 GB.",
    ),
    # -- MoE experts: the bulk of the weights ---------------------------------
    # gate/up are the least sensitive (Unsloth spends 3.4 bpw here), down gets more.
    # Sharded on the *intermediate* axis, not the expert axis. The expert axis
    # was chosen because `sparse_matmul` zero-fills its whole [1, E, M, N] output
    # (invariant 27/40), and 512 experts a device made that 84 MB a layer against
    # 21. The wide gather path retired `sparse_matmul` from decode entirely, and
    # what the expert axis costs there is a fixed `k_sel` of 10: the top-10 lands
    # unevenly on four devices, the expected count is 2.5, and three quarters of
    # every gather and both matmuls is padding that carries zero weight.
    #
    # Giving every device all 512 experts at a quarter of the intermediate width
    # removes the worst case instead of paying for it -- same bytes resident,
    # same arithmetic, and the gathered part is now all useful.
    # `scripts/stage8_expert_shard_axis.py` measures 20.70 -> 9.88 ms a token
    # over the gather and both matmuls, 2.10x, at a *global* k_sel of 16 that
    # `kept_expert_census.py` shows drops nothing (the tie admission keeps at
    # most 14 experts over 2256 real routing decisions).
    #
    # Prefill still routes through `sparse_matmul`, and there the down
    # projection's output does grow four times; measured in handoff 15.
    TensorPlan(
        r"^blk\.\d+\.ffn_gate_exps\.weight$", Residency.DEVICE, "bfloat4_b",
        Shard.EXPERT_COLUMN,
        "IQ3_S upstream -- lowest-sensitivity tensor in the model",
    ),
    TensorPlan(
        r"^blk\.\d+\.ffn_up_exps\.weight$", Residency.DEVICE, "bfloat4_b",
        Shard.EXPERT_COLUMN,
        "IQ3_S upstream",
    ),
    TensorPlan(
        r"^blk\.\d+\.ffn_down_exps\.weight$", Residency.DEVICE, "bfloat8_b",
        Shard.EXPERT_ROW,
        "IQ4_NL upstream and Q8_0 for layers 0-4; down_proj carries more error weight",
    ),
    # -- routers stay in f32: a wrong expert is not a small perturbation -------
    TensorPlan(r"^blk\.\d+\.ffn_gate_inp\.weight$", Residency.DEVICE, "float32", Shard.REPLICATE,
               "F32 upstream; top-k selection must not wobble"),
    TensorPlan(r"^blk\.\d+\.ffn_gate_inp_shexp\.weight$", Residency.DEVICE, "float32", Shard.REPLICATE),
    # -- QSA indexer: selects *which* tokens are attended to -------------------
    TensorPlan(r"^blk\.\d+\.indexer\..*\.weight$", Residency.DEVICE, "bfloat16", Shard.REPLICATE,
               "BF16 upstream; an error here changes the token set, not just the value"),
    # -- shared expert ---------------------------------------------------------
    # Replicated it read 266 MB a device a token and cost no collective -- but
    # "avoids a collective per layer" was measuring against the wrong baseline:
    # `_moe_block` **already** all-reduces the routed sum on the very next line,
    # and a sharded shared expert's partial adds into that one. Same collective
    # count, a quarter of the bytes. `gate|up` split on their output so each
    # device owns 160 of the 640 intermediate; `down` split on its contraction
    # so its output is a partial, which is what rides the reduce.
    TensorPlan(r"^blk\.\d+\.ffn_(gate|up)_shexp\.weight$", Residency.DEVICE, "bfloat8_b",
               Shard.COLUMN, "160 of 640 intermediate a device"),
    TensorPlan(r"^blk\.\d+\.ffn_down_shexp\.weight$", Residency.DEVICE, "bfloat8_b",
               Shard.ROW, "reduction split; its partial joins the routed one"),
    # -- full attention --------------------------------------------------------
    TensorPlan(r"^blk\.\d+\.attn_(q|k|v|output)\.weight$", Residency.DEVICE, "bfloat8_b",
               Shard.REPLICATE, "0.6 GB over 12 layers; replicated to drop an all-reduce per layer"),
    # -- linear attention (Gated DeltaNet) ------------------------------------
    # 48 V heads / 16 K heads both divide by 4, so head-sharding is exact.
    # Head-sharded. Replicating these was right when the decode path was purely
    # dispatch-bound (it halved the collectives), but at batch 32 the DeltaNet is
    # ~65% of the step and every device was redundantly computing all 48 heads
    # over a 3.6 GB replicated recurrent state. Sharding 48 V-heads / 16 K-heads
    # four ways cuts that compute and state 4x, at one all-reduce per linear
    # layer -- and the state shrink is what lets the batch grow past 32.
    TensorPlan(r"^blk\.\d+\.attn_qkv\.weight$", Residency.DEVICE, "bfloat8_b",
               Shard.HEAD_QKV_COLUMN, "10240 channels = q|k|v, split per head"),
    TensorPlan(r"^blk\.\d+\.ssm_conv1d\.weight$", Residency.DEVICE, "float32",
               Shard.HEAD_QKV_ROW, "same 10240 channels, on dim -2"),
    TensorPlan(r"^blk\.\d+\.attn_gate\.weight$", Residency.DEVICE, "bfloat8_b", Shard.COLUMN),
    TensorPlan(r"^blk\.\d+\.ssm_(alpha|beta)\.weight$", Residency.DEVICE, "float32", Shard.COLUMN),
    TensorPlan(r"^blk\.\d+\.ssm_a$", Residency.DEVICE, "float32", Shard.COLUMN),
    TensorPlan(r"^blk\.\d+\.ssm_dt\.bias$", Residency.DEVICE, "float32", Shard.COLUMN),
    TensorPlan(r"^blk\.\d+\.ssm_out\.weight$", Residency.DEVICE, "bfloat8_b", Shard.ROW),
    TensorPlan(r"^blk\.\d+\.ssm_norm\.weight$", Residency.DEVICE, "float32",
               Shard.REPLICATE, "per head_dim, identical across heads"),
    TensorPlan(r"^blk\.\d+\.attn_(q|k)_norm\.weight$", Residency.DEVICE, "float32", Shard.REPLICATE,
               "per-head-dim norms, 256 elements each"),
    # -- hyper-connections and norms -------------------------------------------
    # `down` is [10240, 320] -- 10240 of reduction against 10 output tiles, which
    # is far too narrow to fill the grid: replicated it reads 3.48 MB a call at
    # 0.1209 ms, and it runs 96 times a token. Splitting the reduction four ways
    # makes each device read 0.87 MB (0.0324 ms) and the 320-wide partial costs
    # one all_reduce, which is latency-bound at 0.0397 ms whatever its width --
    # 0.0445 against 0.1209, so 11.61 ms a token becomes 4.27
    # (`hc_down_shard_check.py`). Same products, summed in a different order.
    TensorPlan(r"^(blk\.\d+\.hc_\w+|output_hc)_down\.weight$", Residency.DEVICE, "bfloat8_b",
               Shard.ROW, "reduction split 4 ways; consumer all-reduces"),
    # `up` is the transpose, [320, 10240], and a 10240-wide output already fills
    # the grid -- 0.0157 ms a call. Sharding its 320 of reduction would starve it
    # and buy an all_reduce that costs more than the matmul. Left replicated.
    TensorPlan(r"^(blk\.\d+\.hc_\w+|output_hc)_up\.weight$", Residency.DEVICE, "bfloat8_b",
               Shard.REPLICATE),
    TensorPlan(r"^(blk\.\d+\.hc_\w+|output_hc)_(norm|inject)\.weight$", Residency.DEVICE, "float32",
               Shard.REPLICATE),
    # -- PLE projections --------------------------------------------------------
    TensorPlan(r"^blk\.\d+\.ple_(key|value)\.weight$", Residency.DEVICE, "bfloat8_b", Shard.REPLICATE),
    TensorPlan(r"^blk\.\d+\.ple_(norm_\w+|conv1d)\.weight$", Residency.DEVICE, "float32", Shard.REPLICATE),
    # -- embeddings and head ----------------------------------------------------
    TensorPlan(r"^token_embd\.weight$", Residency.HOST, "source", Shard.REPLICATE,
               "a gather like the n-gram table; only the prompt's rows are needed"),
    TensorPlan(r"^output\.weight$", Residency.DEVICE, "bfloat8_b", Shard.COLUMN,
               "shard the vocabulary, all-gather the logits"),
)

_COMPILED = tuple((re.compile(p.pattern), p) for p in PLAN)

# Bytes per element for the device formats. Block-float types carry one shared
# exponent per 16 datums, hence the +1/16 byte.
BYTES_PER_ELEMENT: dict[str, float] = {
    "float32": 4.0,
    "bfloat16": 2.0,
    "bfloat8_b": 1.0 + 1.0 / 16.0,
    "bfloat4_b": 0.5 + 1.0 / 16.0,
}


def plan_for(name: str) -> TensorPlan:
    for regex, entry in _COMPILED:
        if regex.match(name):
            return entry
    raise KeyError(f"no residency plan covers tensor {name!r}")


@dataclass(slots=True)
class Budget:
    device_bytes: float = 0.0
    host_bytes: float = 0.0
    per_device_bytes: float = 0.0
    unplanned: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.unplanned is None:
            self.unplanned = []


def compute_budget(tensors: dict, num_devices: int = 4) -> Budget:
    """`tensors` maps name -> object with `.n_elements` and `.nbytes`."""
    budget = Budget()
    for name, info in tensors.items():
        try:
            entry = plan_for(name)
        except KeyError:
            budget.unplanned.append(name)
            continue
        # "source" means the tensor is left in its GGUF quantisation and read
        # through the mmap, so it costs its on-disk size and nothing more.
        nbytes = (
            info.nbytes
            if entry.dtype == "source"
            else info.n_elements * BYTES_PER_ELEMENT[entry.dtype]
        )
        if entry.residency is Residency.HOST:
            budget.host_bytes += nbytes
            continue
        budget.device_bytes += nbytes
        if entry.shard is Shard.REPLICATE:
            budget.per_device_bytes += nbytes
        else:
            budget.per_device_bytes += nbytes / num_devices
    return budget
