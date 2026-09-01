"""Convert a GGUF checkpoint into a ttnn `.tensorbin` weight cache.

Dequantising 93 GB of IQ3_S/IQ4_NL and re-encoding it to block float takes
roughly an hour, so it is done once. `ttnn.load_tensor` then reads it back
losslessly and effectively instantly (verified: exact byte sizes, bit-identical
round-trip, sub-millisecond loads).

Replicated tensors get one file; **sharded tensors get one file per device**,
split here with `torch.chunk` before quantisation. That split placement is not
cosmetic -- it is worth ~95x on model load:

    host split via distribute_tensor   18.09 s   (26 MB/s)
    to_device                           0.15 s   (3220 MB/s)
    from_host_shards + to_device        0.08 s   (24188 MB/s)

`ttnn.distribute_tensor(..., ShardTensorToMesh)` splits a tiled block-float
tensor on the host at 26 MB/s, which for 88.8 GB of expert weights is ~40
minutes every time the model loads. Pre-splitting turns loading into
`load_tensor` (mmap, free) plus `from_host_shards`, and the whole model loads in
seconds.

Quantising each shard separately is equivalent to quantising the whole tensor
and then splitting, because every shard boundary here is a multiple of 16 -- the
block-float group size -- along the last dimension.

Only the expert stacks and the LM head are sharded; see plan.py for why the
dense tensors are replicated instead.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..gguf.reader import GGUFModel
from ..reference.config import Qwen4ExpConfig
from ..reference.weights import WeightStore
from .plan import Residency, Shard, plan_for

MANIFEST = "manifest.json"

# Which axis of the *device layout* each shard mode splits.
SHARD_DIM = {
    Shard.REPLICATE: None,
    Shard.COLUMN: -1,
    Shard.ROW: -2,
    Shard.EXPERT_COLUMN: -1,
    Shard.EXPERT_ROW: -2,
    Shard.HEAD_QKV_COLUMN: -1,
    Shard.HEAD_QKV_ROW: -2,
}
HEAD_QKV = {Shard.HEAD_QKV_COLUMN, Shard.HEAD_QKV_ROW}
# Whether the consumer must all-reduce after using the tensor.
NEEDS_ALL_REDUCE = {Shard.ROW, Shard.EXPERT_ROW}


def split_qkv_channels(t, dim: int, n_dev: int, key_dim: int, value_dim: int):
    """Split a 10240-wide q|k|v channel axis by head, not flat.

    The axis is [q(2*key_dim/2) | k | v]; chunking it evenly at 2560 would cut
    through the q/k boundary and hand each device a mix of one head's queries
    and another's keys -- correct shapes, silently wrong pairing. Each part is
    chunked separately and re-concatenated per device, so device i gets
    q-heads i, k-heads i and v-heads i.
    """
    import torch

    q, k, v = torch.split(t, [key_dim, key_dim, value_dim], dim=dim)
    qs = torch.chunk(q, n_dev, dim=dim)
    ks = torch.chunk(k, n_dev, dim=dim)
    vs = torch.chunk(v, n_dev, dim=dim)
    return [torch.cat([qs[i], ks[i], vs[i]], dim=dim).contiguous() for i in range(n_dev)]


@dataclass(slots=True)
class ConvertStats:
    tensors: int = 0
    bytes_written: int = 0
    skipped_host: int = 0
    entries: dict = field(default_factory=dict)


# The hyper-connection gate computes silu(down(x)/hc) and 2*sigmoid(inject(x)/hc).
# Dividing by hc_count is a power of two, so folding it into the weights is exact
# in block float -- it only shifts the shared exponent -- and removes two scalar
# multiplies from a path that runs 96 times per token (0.41 ms per call at B=64).
PRESCALE = {r"^(blk\.\d+\.hc_\w+|output_hc)_(down|inject)\.weight$": 0.25}


def prescale_factor(name: str) -> float:
    for pattern, factor in PRESCALE.items():
        if re.match(pattern, name):
            return factor
    return 1.0


def device_layout(name: str, t: torch.Tensor) -> torch.Tensor:
    """Reshape a GGUF tensor into the layout its device op consumes."""
    # Expert stacks: GGUF (E, out, in) -> [1, E, K, N] for sparse_matmul.
    # This one transpose serves gate/up (out=640) and down (out=2560) alike,
    # because both become (E, in, out) = [E, K, N].
    if re.match(r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$", name):
        return t.transpose(1, 2).unsqueeze(0).contiguous()
    # Depthwise conv filters stay (channels, kernel_width).
    if re.search(r"(ssm_conv1d|ple_conv1d)\.weight$", name):
        return t.reshape(1, 1, *t.shape).contiguous()
    # The shared-expert sigmoid gate is 1-D but is a *linear weight*
    # (hidden_size -> 1), so it needs a column layout, not a row. Every other
    # 1-D tensor here is applied elementwise (norm gamma, ssm_a, dt_bias) and
    # wants the trailing-dim row form.
    if re.match(r"^blk\.\d+\.ffn_gate_inp_shexp\.weight$", name):
        return t.reshape(1, 1, -1, 1).contiguous()
    # 1-D norms / decays / biases as a trailing-dim vector.
    if t.ndim == 1:
        return t.reshape(1, 1, 1, -1).contiguous()
    # Everything else is a linear weight: GGUF (out, in) -> (in, out).
    return t.transpose(-1, -2).reshape(1, 1, t.shape[-1], t.shape[-2]).contiguous()


def convert(
    model_dir: str | Path,
    out_dir: str | Path,
    n_dev: int = 4,
    only: str | None = None,
    progress=print,
) -> ConvertStats:
    import ttnn

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gguf = GGUFModel.from_dir(model_dir)
    cfg = Qwen4ExpConfig.from_gguf(gguf.metadata)
    cfg.validate()
    # Small caches: this walks every tensor exactly once, so caching only wastes RAM.
    store = WeightStore(gguf, cache_bytes=1 << 28, row_cache_bytes=1 << 28)

    dtypes = {
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": ttnn.bfloat4_b,
        "bfloat16": ttnn.bfloat16,
        "float32": ttnn.float32,
    }
    stats = ConvertStats()
    manifest_path = out / MANIFEST
    if manifest_path.exists():
        stats.entries = json.loads(manifest_path.read_text()).get("tensors", {})

    names = sorted(gguf.tensors)
    if only:
        names = [n for n in names if re.search(only, n)]
    started = time.time()

    for idx, name in enumerate(names):
        entry = plan_for(name)
        if entry.residency is Residency.HOST:
            stats.skipped_host += 1
            continue

        shard = entry.shard
        dim = SHARD_DIM[shard]
        if dim is None:
            paths = [out / f"{name}.tensorbin"]
        else:
            paths = [out / f"{name}.dev{i}.tensorbin" for i in range(n_dev)]
        record = {
            "files": [p.name for p in paths],
            "dtype": entry.dtype,
            "shard": shard.name,
            "shard_dim": dim,
            "all_reduce": shard in NEEDS_ALL_REDUCE,
        }
        if all(p.exists() for p in paths) and name in stats.entries:
            continue

        info = store.info(name)
        # Expert stacks are far too big to hold twice; pull them as one row range.
        t = (
            store.get_row_range(name, 0, info.torch_shape[0])
            if info.n_elements >= (1 << 26)
            else store.get(name)
        )
        scale = prescale_factor(name)
        if scale != 1.0:
            t = t * scale
        t = device_layout(name, t)
        # No numpy pre-quantisation here: `from_torch(dtype=bfloat*_b)` quantises
        # to the same grid, and `tt/blockfloat.py` was verified elementwise
        # identical to it -- so pre-rounding is pure redundant work (it was
        # measured at ~60% of the per-tensor cost). The numpy quantiser exists to
        # simulate device precision inside the CPU reference, not to feed this.
        if dim is None:
            pieces = [t]
        elif shard in HEAD_QKV:
            pieces = split_qkv_channels(t, dim, n_dev, cfg.linear_key_dim, cfg.linear_value_dim)
        else:
            pieces = list(torch.chunk(t, n_dev, dim=dim))
        for piece, path in zip(pieces, paths):
            host = ttnn.from_torch(piece.contiguous(), dtype=dtypes[entry.dtype], layout=ttnn.TILE_LAYOUT)
            ttnn.dump_tensor(str(path), host)
            del host
        del t, pieces

        record["shape"] = list(store.info(name).torch_shape)
        record["bytes"] = sum(p.stat().st_size for p in paths)
        stats.entries[name] = record
        stats.bytes_written += record["bytes"]
        stats.tensors += 1
        store.clear_cache()

        if progress and stats.tensors % 20 == 0:
            done = stats.tensors
            elapsed = time.time() - started
            progress(
                f"[{idx + 1}/{len(names)}] {done} tensors, {stats.bytes_written / 1e9:.2f} GB, "
                f"{elapsed:.0f}s elapsed"
            )
            manifest_path.write_text(json.dumps({"n_dev": n_dev, "tensors": stats.entries}, indent=1))

    manifest_path.write_text(json.dumps({"n_dev": n_dev, "tensors": stats.entries}, indent=1))
    gguf.close()
    return stats
