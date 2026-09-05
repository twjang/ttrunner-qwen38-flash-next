"""Can a small ttnn op be made to launch on fewer cores?

    uv run python scripts/dev/small_op_core_grid.py

`dispatch_floor.py` prices a launch by its core count: a `generic_op` touching
one page is 2.06 us on one core, 3.15 on thirty-two and 5.81 on a hundred and
ten -- and `ttnn.multiply` on a *one tile* tensor is 5.78, which is the
hundred-and-ten number. So ttnn takes the whole grid whatever the tensor, and
the model issues ~2450 elementwise ops a step, most of them on a handful of
tiles.

3.7 us x 2450 is 9 ms of a 49 ms step, spent launching cores that have nothing
to do.

Two ways to not do that, and this measures both: pin the operands into an L1
shard on a small core range, or pass a `core_grid`. Whichever works, the fix is
mechanical rather than a kernel.
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
torch.manual_seed(0)
grid = mesh.compute_with_storage_grid_size()


def timed(fn, reps=64, iters=8):
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
    return best / reps * 1e6


SHAPES = [("1 tile   [1,1,32,32]", (1, 1, 32, 32)),
          ("10 tiles [1,1,32,320]", (1, 1, 32, 320)),
          ("80 tiles [1,1,32,2560]", (1, 1, 32, 2560))]

for label, shape in SHAPES:
    a = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    base = timed(lambda a=a: ttnn.multiply(a, a))
    print(f"RESULT {label}: interleaved {base:6.2f} us", flush=True)

    nt = (shape[2] // 32) * (shape[3] // 32)
    for n_cores in (1, 2, 4, 8):
        if nt % n_cores:
            continue
        try:
            crs = ttnn.num_cores_to_corerangeset(n_cores, grid, True)
            spec = ttnn.ShardSpec(crs, [shape[2], shape[3] // n_cores],
                                  ttnn.ShardOrientation.ROW_MAJOR)
            cfg_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                                       ttnn.BufferType.L1, spec)
            sh = ttnn.to_memory_config(a, cfg_l1)
            us = timed(lambda sh=sh: ttnn.multiply(sh, sh))
            ok = ttnn.to_torch(ttnn.multiply(sh, sh),
                               mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
            ref = ttnn.to_torch(ttnn.multiply(a, a),
                                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
            d = (ok - ref).abs().max().item()
            print(f"RESULT    L1 sharded, {n_cores:3d} cores {us:6.2f} us "
                  f"{base/us:5.2f}x  diff {d:.1e}", flush=True)
            ttnn.deallocate(sh)
        except Exception as exc:                                      # noqa: BLE001
            print(f"RESULT    L1 sharded, {n_cores:3d} cores: {type(exc).__name__}: "
                  f"{(str(exc) or repr(exc)).splitlines()[0][:90]}", flush=True)
    # and the cost of getting it there
    try:
        crs = ttnn.num_cores_to_corerangeset(1, grid, True)
        spec = ttnn.ShardSpec(crs, [shape[2], shape[3]], ttnn.ShardOrientation.ROW_MAJOR)
        cfg_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                                   ttnn.BufferType.L1, spec)
        conv = timed(lambda a=a, c=cfg_l1: ttnn.to_memory_config(a, c))
        print(f"RESULT    (to_memory_config onto 1 core: {conv:6.2f} us)", flush=True)
    except Exception as exc:                                          # noqa: BLE001
        print(f"RESULT    (to_memory_config: {type(exc).__name__})", flush=True)
    ttnn.deallocate(a)

ttnn.close_mesh_device(mesh)
