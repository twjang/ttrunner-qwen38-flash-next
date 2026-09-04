"""Does keeping the gathered expert weights in L1 remove the copy's cost?

    uv run python scripts/stage6_l1_weights.py

The MoE is 33.28 ms of an 82.35 ms step and its two gathers are 13.19 of that
(`decode_ablation_check.py w:gather`). The gather exists only to put the
selected experts side by side so one dense `ttnn.linear` can read them, and it
pays for that twice: it reads 24.6 MB of expert weights from DRAM and writes
24.6 MB back, and then the linear reads those 24.6 MB *again*. 73.8 MB moved
where 24.6 MB is the actual requirement -- three times the roofline, which is
exactly the "unnecessary memory copy" the goal names.

The write is the removable half. If the gather lands its tiles in **L1** rather
than DRAM, the DRAM traffic is one read of the weights and nothing else, and the
matmul reads from L1 at core-local speed. Nothing has to be written to make that
work *if* `ttnn.linear` will take a sharded in1 -- which is what this measures,
before any kernel is touched.

Three questions, in order, because each only matters if the previous one holds:
  1. can a bfloat4_b weight even be L1 width-sharded at these shapes?
  2. does `ttnn.linear` accept it as in1 at M=1, and give the same answer?
  3. is it faster than the DRAM path, and by how much?

Shapes are the model's own: gate|up gathered is [2560, 10*1280] and down is
[10*640, 2560], both bfloat4_b, at M=1.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import ttnn

# (label, K, N, dtype) -- the two gathered shapes wide_expert_ffn feeds to linear
SHAPES = [
    ("gate|up  [2560, 12800]", 2560, 10 * 1280, ttnn.bfloat4_b),
    ("down     [6400, 2560]", 10 * 640, 2560, ttnn.bfloat4_b),
]


def l1_width_sharded(t, grid):
    """Width-shard `t`'s last dim across the grid, in L1."""
    n_cores = grid.x * grid.y
    nt = t.shape[-1] // 32
    # every core must own a whole number of tiles
    cores = n_cores
    while cores > 1 and nt % cores:
        cores -= 1
    if cores < 2:
        raise RuntimeError(f"N={t.shape[-1]} does not divide over the grid")
    per = t.shape[-1] // cores
    rows, cols = (cores + grid.x - 1) // grid.x, grid.x
    crs = ttnn.num_cores_to_corerangeset(cores, grid, True)
    spec = ttnn.ShardSpec(crs, [t.shape[-2], per], ttnn.ShardOrientation.ROW_MAJOR)
    cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
    return ttnn.to_memory_config(t, cfg), cores, per


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dev"))
    from _device_model import open_model                              # noqa: E402

    mesh, cfg, m = open_model(max_seq_len=512)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    torch.manual_seed(0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        print(f"RESULT grid {grid.x}x{grid.y} = {grid.x * grid.y} cores", flush=True)
        for attr in ("l1_size_per_core", "l1_size"):
            fn = getattr(mesh, attr, None)
            if fn is not None:
                print(f"RESULT L1 per core {fn()} B", flush=True)
                break

        for label, K, N, wdt in SHAPES:
            x = ttnn.from_torch(torch.randn(1, 1, 32, K) * 0.05, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.02, dtype=wdt,
                                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            per_core_bytes = K * (N // (grid.x * grid.y)) // 2      # bfloat4_b ~ 0.5 B
            print(f"RESULT {label}: weight {K*N//2/2**20:.1f} MiB, "
                  f"~{per_core_bytes/1024:.0f} KiB a core if spread over all of them",
                  flush=True)

            try:
                w_l1, cores, per = l1_width_sharded(w, grid)
            except Exception as exc:                                  # noqa: BLE001
                print(f"RESULT   L1 shard rejected: {type(exc).__name__}: "
                      f"{(str(exc) or repr(exc)).splitlines()[0][:200]}", flush=True)
                continue
            print(f"RESULT   sharded over {cores} cores, {per} columns each "
                  f"({K*per//2/1024:.0f} KiB a core)", flush=True)

            ref = ttnn.linear(x, w)
            try:
                got = ttnn.linear(x, w_l1)
            except Exception as exc:                                  # noqa: BLE001
                print(f"RESULT   linear(L1-sharded in1) rejected: {type(exc).__name__}: "
                      f"{(str(exc) or repr(exc)).splitlines()[0][:200]}", flush=True)
                continue
            a = ttnn.to_torch(ref, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
            b = ttnn.to_torch(got, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
            err = (a.to(torch.float64) - b.to(torch.float64)).abs().max().item()
            scale = a.abs().max().item()
            print(f"RESULT   same answer? max abs diff {err:.3e} "
                  f"(rel {err/max(scale,1e-30):.2e})", flush=True)

            def timed(fn, tag, reps=30, iters=8):
                for _ in range(2):
                    fn()
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                for _ in range(reps):
                    fn()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                best = float("inf")
                for _ in range(iters):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    best = min(best, time.perf_counter() - t0)
                ttnn.release_trace(mesh, tid)
                us = best / reps * 1e6
                print(f"RESULT   {tag:34s} {us:8.2f} us  "
                      f"({48 * us / 1000:6.2f} ms over 48 layers)", flush=True)
                return us

            d = timed(lambda: ttnn.linear(x, w), "linear, weight in DRAM")
            l = timed(lambda: ttnn.linear(x, w_l1), "linear, weight in L1 sharded")
            print(f"RESULT   -> {d / l:.2f}x", flush=True)
            # And what it costs to *put* it there, which the gather would do instead
            c = timed(lambda: ttnn.to_memory_config(w, w_l1.memory_config()),
                      "DRAM -> L1 copy (the gather's job)")
            print(f"RESULT   round trip {c + l:.2f} us against {d:.2f} us "
                  f"for DRAM alone", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
