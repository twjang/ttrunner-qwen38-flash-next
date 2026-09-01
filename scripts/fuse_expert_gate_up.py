"""Build fused gate|up expert weights, exactly, from the existing device cache.

Why: the MoE's cost is a per-call floor, not the work it discards -- the block
costs 4.497 ms at M=1 where it wastes nothing -- so issuing one sparse_matmul
instead of two is worth 1.84x on the gate+up pair (7.749 -> 4.210 ms at M=64).

Why not `ttnn.concat` at load: it requantises bfloat4_b (measured 0.0547 max
error), which changed generated tokens. Dequantising to float and quantising the
concatenation once is exact (measured 0.0), because the per-device width 160 is a
whole number of 16-element blocks, so the shared exponents line up.

Writes new files and leaves the originals untouched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import ttnn

CACHE = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/twjang/models/qwen38-tt-cache")
MANIFEST = CACHE / "manifest.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    tensors = manifest["tensors"]
    n_dev = manifest["n_dev"]
    layers = sorted(
        int(n.split(".")[1]) for n in tensors if n.endswith("ffn_gate_exps.weight")
    )
    print(f"fusing {len(layers)} layers x {n_dev} devices", flush=True)
    started = time.time()
    written = 0

    for layer in layers:
        name = f"blk.{layer}.ffn_gateup_exps.weight"
        gate_rec = tensors[f"blk.{layer}.ffn_gate_exps.weight"]
        up_rec = tensors[f"blk.{layer}.ffn_up_exps.weight"]
        paths = [CACHE / f"{name}.dev{i}.tensorbin" for i in range(n_dev)]
        if all(p.exists() for p in paths) and name in tensors:
            continue

        for i, (gf, uf, path) in enumerate(zip(gate_rec["files"], up_rec["files"], paths)):
            # load one device's shard at a time: the full stack is 3.35 GB in fp32
            g = ttnn.to_torch(ttnn.load_tensor(str(CACHE / gf))).float()
            u = ttnn.to_torch(ttnn.load_tensor(str(CACHE / uf))).float()
            fused = torch.cat([g, u], dim=-1).contiguous()
            del g, u
            host = ttnn.from_torch(fused, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
            ttnn.dump_tensor(str(path), host)
            del fused, host

        tensors[name] = {
            "files": [p.name for p in paths],
            "dtype": "bfloat4_b",
            "shard": gate_rec["shard"],
            "shard_dim": gate_rec["shard_dim"],
            "all_reduce": gate_rec["all_reduce"],
            "shape": [gate_rec["shape"][0], gate_rec["shape"][1], gate_rec["shape"][2] * 2]
            if len(gate_rec["shape"]) == 3
            else gate_rec["shape"],
            "bytes": sum(p.stat().st_size for p in paths),
        }
        written += 1
        elapsed = time.time() - started
        print(
            f"  blk.{layer}: {tensors[name]['bytes']/1e9:.2f} GB  "
            f"[{written}/{len(layers)}] {elapsed:.0f}s",
            flush=True,
        )
        MANIFEST.write_text(json.dumps(manifest))

    print(f"done: {written} layers in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
