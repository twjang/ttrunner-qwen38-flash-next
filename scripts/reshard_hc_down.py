"""Re-convert the `hc_*_down` weights as a reduction shard, four processes.

    uv run python scripts/reshard_hc_down.py

`hc_down` moved from `Shard.REPLICATE` to `Shard.ROW` in plan.py: its 10240 of
reduction is now split across the four devices and `gated_residual_mix`
all-reduces the 320-wide partial before the silu. Replicated it read 3.48 MB a
call at 0.1209 ms and ran 96 times a token; split it is 0.0445 ms including the
collective, so 11.61 ms a token becomes 4.27 (`hc_down_shard_check.py`).

97 tensors, 338 MB -- nothing like the 130 GB expert reshard. This needs no
device at all: `convert()` builds host tensors with `ttnn.from_torch` and writes
them with `dump_tensor`, so four processes over disjoint layer ranges just work.
Each writes its own manifest, because `convert()` loads the manifest whole and
writes it back whole and the last writer would otherwise erase the others; the
merge at the end folds them into the real one.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

GGUF = os.environ.get("TT_GGUF_DIR", str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS"))
CACHE = os.environ.get("TT_CACHE_DIR", str(Path.home() / "models/qwen38-tt-cache"))
# One process, not four. `import ttnn` opens the UMD cluster and takes
# `CHIP_IN_USE_0_PCIe`, and `TT_METAL_VISIBLE_DEVICES=n` does not separate the
# workers: the visible chip is renumbered to 0, so every worker queues on the
# same lock name and three of them wait forever. This is 97 tensors and 338 MB,
# nothing like the 130 GB expert reshard, so serial is the right trade rather
# than a fight with the lock.
WORKERS = 1
LAYERS = 48


def shard_pattern(worker: int) -> str:
    """The `only=` regex for this worker's disjoint slice of the layers."""
    lo = worker * LAYERS // WORKERS
    hi = (worker + 1) * LAYERS // WORKERS
    layers = "|".join(str(i) for i in range(lo, hi))
    pat = rf"^blk\.({layers})\.hc_\w+_down\.weight$"
    if worker == 0:                      # the final mixing block rides with 0
        pat = rf"({pat[1:]}|^output_hc_down\.weight$)"
    return pat


def run(worker: int) -> tuple[int, int, float]:
    # `convert()` imports ttnn, and importing ttnn opens the UMD cluster -- which
    # takes `CHIP_IN_USE_0_PCIe` on *every* chip. Four workers then queue behind
    # one lock and three of them wait forever ("Waiting for lock
    # 'CHIP_IN_USE_0_PCIe' which is currently held by ..."), which is what an
    # earlier run of this script did. Pinning each worker to its own chip gives
    # each its own lock. The conversion itself never touches the device: the
    # shards are `torch.chunk` and `dump_tensor` writes host tensors, so a worker
    # still writes all four device shards for the tensors it owns.
    os.environ["TT_METAL_VISIBLE_DEVICES"] = str(worker)

    from ttrunner_qwen38_flash_next.tt.convert import convert

    t0 = time.time()
    stats = convert(
        GGUF, CACHE, n_dev=4,
        only=shard_pattern(worker),
        force=True,                      # the files exist and are now wrong
        manifest_name=f"manifest.hc{worker}.json",
        progress=None,
    )
    return worker, stats.tensors, stats.bytes_written, time.time() - t0


def main() -> None:
    print(f"resharding hc_*_down across {WORKERS} processes", flush=True)
    for w in range(WORKERS):
        print(f"  worker {w}: {shard_pattern(w)}", flush=True)

    with mp.get_context("spawn").Pool(WORKERS) as pool:
        results = pool.map(run, range(WORKERS))

    total_t, total_b = 0, 0
    for w, n, b, secs in results:
        print(f"  worker {w}: {n} tensors, {b / 1e6:.0f} MB, {secs:.0f}s", flush=True)
        total_t += n
        total_b += b
    print(f"done: {total_t} tensors, {total_b / 1e6:.0f} MB", flush=True)

    # Fold the per-worker manifests into the real one.
    cache = Path(CACHE)
    main_path = cache / "manifest.json"
    merged = json.loads(main_path.read_text())
    for w in range(WORKERS):
        part = cache / f"manifest.hc{w}.json"
        if not part.exists():
            print(f"  WARNING no manifest from worker {w}", flush=True)
            continue
        merged["tensors"].update(json.loads(part.read_text()).get("tensors", {}))
        part.unlink()
    main_path.write_text(json.dumps(merged, indent=1))

    rows = [(k, v) for k, v in merged["tensors"].items() if k.endswith("_down.weight")
            and ("hc_" in k or k.startswith("output_hc"))]
    bad = [k for k, v in rows if v.get("shard") != "ROW"]
    print(f"manifest merged: {len(rows)} hc_down entries, "
          f"{'all ROW' if not bad else f'STILL WRONG: {bad[:4]}'}", flush=True)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
