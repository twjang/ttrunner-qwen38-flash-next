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
    TensorPlan(
        r"^blk\.\d+\.ffn_gate_exps\.weight$", Residency.DEVICE, "bfloat4_b", Shard.EXPERT_COLUMN,
        "IQ3_S upstream -- lowest-sensitivity tensor in the model",
    ),
    TensorPlan(
        r"^blk\.\d+\.ffn_up_exps\.weight$", Residency.DEVICE, "bfloat4_b", Shard.EXPERT_COLUMN,
        "IQ3_S upstream",
    ),
    TensorPlan(
        r"^blk\.\d+\.ffn_down_exps\.weight$", Residency.DEVICE, "bfloat8_b", Shard.EXPERT_ROW,
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
    TensorPlan(r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$", Residency.DEVICE, "bfloat8_b",
               Shard.REPLICATE, "0.9 GB total; replicating avoids a collective per layer"),
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
    TensorPlan(r"^(blk\.\d+\.hc_\w+|output_hc)_(down|up)\.weight$", Residency.DEVICE, "bfloat8_b",
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
