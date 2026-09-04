"""Load a converted `.tensorbin` weight cache onto the mesh."""

from __future__ import annotations

import json
from pathlib import Path

import ttnn

from .convert import MANIFEST


class TTWeights:
    """Lazily loads device tensors from the cache, keyed by GGUF tensor name."""

    def __init__(
        self, cache_dir: str | Path, mesh, preload: bool = False, fused_experts: bool = False
    ):
        self.dir = Path(cache_dir)
        self.mesh = mesh
        manifest = json.loads((self.dir / MANIFEST).read_text())
        self.n_dev = manifest["n_dev"]
        self.entries: dict[str, dict] = manifest["tensors"]
        self._cache: dict[str, ttnn.Tensor] = {}
        if preload:
            # gate|up and the fused gateup are two spellings of the same weights,
            # and holding both overflows DRAM by ~40 GB ("Not enough space to
            # allocate 235929600 B"). Load whichever set will actually be read.
            redundant = ("ffn_gate_exps", "ffn_up_exps") if fused_experts else ("ffn_gateup_exps",)
            for name in self.entries:
                if any(r in name for r in redundant):
                    continue
                self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def get(self, name: str) -> ttnn.Tensor:
        hit = self._cache.get(name)
        if hit is not None:
            return hit
        try:
            record = self.entries[name]
        except KeyError:
            raise KeyError(f"{name!r} is not in the weight cache at {self.dir}") from None

        files = record.get("files") or [record["file"]]
        dim = record["shard_dim"]
        if dim is None:
            # load_tensor(device=mesh) gives every device an identical copy, and
            # is fast (measured 4256 MB/s)
            tensor = ttnn.load_tensor(str(self.dir / files[0]), device=self.mesh)
        else:
            # Shards were pre-split at conversion time. Composing them with
            # from_host_shards takes 0.08 s where distribute_tensor's host-side
            # split took 18.09 s for the same 472 MB tensor -- ~226x.
            shards = [ttnn.load_tensor(str(self.dir / f)) for f in files]
            tensor = ttnn.from_host_shards(shards, self.mesh.shape)
            tensor = ttnn.to_device(tensor, self.mesh)
        self._cache[name] = tensor
        return tensor

    def blk(self, layer: int, suffix: str) -> ttnn.Tensor:
        return self.get(f"blk.{layer}.{suffix}")

    def fused_gate_up(self, layer: int) -> ttnn.Tensor:
        """gate|up experts as one [1, E, K, 2N] tensor, built once and cached.

        The MoE's cost is dominated by a per-call floor rather than by the work it
        discards -- the block costs 4.497 ms at M=1, where it wastes nothing at
        all -- so the lever is issuing fewer sparse_matmuls, not smaller ones.
        gate and up are both bfloat4_b, both EXPERT_COLUMN-sharded and identically
        shaped, so they concatenate into a single call.

        The originals are deallocated once fused: nothing reads them afterwards,
        and keeping all three would add ~105 MB per layer per device. The concat
        itself is transient, so peak use rises only while one layer is being
        built. N = 160 is a whole number of 16-element bfloat4 blocks, so the
        shared exponents line up and the concat does not requantise.
        """
        key = f"blk.{layer}.ffn_gateup_exps.weight"
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if key not in self.entries:
            raise KeyError(
                f"{key!r} is not in the weight cache. Build it with "
                "scripts/fuse_expert_gate_up.py -- fusing on device instead would "
                "requantise the bfloat4_b weights (0.0547 max error) and change "
                "the generated tokens."
            )
        fused = self.get(key)
        # nothing reads the separate halves once the fused tensor is resident, and
        # holding all three would add ~105 MB per layer per device
        for name in (f"blk.{layer}.ffn_gate_exps.weight", f"blk.{layer}.ffn_up_exps.weight"):
            stale = self._cache.pop(name, None)
            if stale is not None:
                ttnn.deallocate(stale)
        return fused

    def needs_all_reduce(self, name: str) -> bool:
        return bool(self.entries[name]["all_reduce"])

    def resident_bytes(self, n_dev: int | None = None) -> int:
        """Bytes held per device (sharded entries counted as their 1/n_dev slice)."""
        n = n_dev or self.n_dev
        total = 0
        for name in self._cache:
            e = self.entries[name]
            total += e["bytes"] if e["shard"] == "REPLICATE" else e["bytes"] / n
        return int(total)
