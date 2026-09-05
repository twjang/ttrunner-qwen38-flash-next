"""Is `ttnn.rms_norm` slow for the same reason the matmul was?

    uv run python scripts/dev/rms_norm_config.py

`grouped_norm_pieces.py` puts 18.17 us of `grouped_rms_norm`'s 36 on the
`rms_norm` itself, on an 80-tile tensor -- three to four times what a wide
elementwise op costs. The matmul turned out to be running on a program config
that read two k-tiles a block (handoff 19); `ttnn` exposes
`LayerNormShardedMultiCoreProgramConfig` too, and it is worth ten minutes to see
whether the same thing is true here before writing a kernel.

The sharded config needs a sharded input, so the sharding cost is measured with
it -- a fair comparison has to include getting the tensor there.
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
H, HC = cfg.hidden_size, cfg.hc_count


def timed(fn, reps=40, iters=8):
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


comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
# the shape `grouped_rms_norm` actually normalises at M = 1
for rows, width in ((HC, H), (1, 2560), (12, 128)):
    xh = torch.randn(1, 1, rows, width) * 0.5
    x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh, mesh_mapper=rep)
    nt = width // 32
    base = timed(lambda x=x: ttnn.rms_norm(x, epsilon=1e-6))
    ref = ttnn.to_torch(ttnn.rms_norm(x, epsilon=1e-6), mesh_composer=comp)[:1]
    # float64 on the operand the device holds -- "differs from the default" is
    # not "wrong" (invariant 57), and a normalisation is the last place to guess.
    x64 = ttnn.to_torch(x, mesh_composer=comp)[:1].to(torch.float64)
    truth = x64 / torch.sqrt((x64 * x64).mean(dim=-1, keepdim=True) + 1e-6)
    sc = max(truth.abs().max().item(), 1e-30)
    e_ref = (ref.to(torch.float64) - truth).abs().max().item() / sc
    print(f"RESULT [1,1,{rows},{width}] ({nt} tiles a row): default "
          f"{base:6.2f}us", flush=True)

    for n_cores in (2, 4, 8, 10, 20, 40, 80):
        if nt % n_cores or n_cores > grid.x * grid.y:
            continue
        bw = nt // n_cores
        for sub in (bw, 4, 2, 1):
            if bw % sub:
                continue
            try:
                crs = ttnn.num_cores_to_corerangeset(n_cores, grid, True)
                spec = ttnn.ShardSpec(crs, [max(rows, 32), width // n_cores],
                                      ttnn.ShardOrientation.ROW_MAJOR)
                mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                                       ttnn.BufferType.L1, spec)
                xs = ttnn.to_memory_config(x, mc)
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=grid,
                    subblock_w=sub, block_h=max(1, (rows + 31) // 32),
                    block_w=bw, inplace=False)
                got = ttnn.to_torch(
                    ttnn.rms_norm(xs, epsilon=1e-6, program_config=pc,
                                  memory_config=mc),
                    mesh_composer=comp)[:1]
                us = timed(lambda xs=xs, pc=pc, mc=mc: ttnn.rms_norm(
                    xs, epsilon=1e-6, program_config=pc, memory_config=mc))
                conv = timed(lambda x=x, mc=mc: ttnn.to_memory_config(x, mc))
                e_got = (got.to(torch.float64) - truth).abs().max().item() / sc
                print(f"RESULT   {n_cores:3d} cores blk_w {bw:3d} sub {sub:2d}: "
                      f"{us:6.2f}us (+{conv:5.2f} to shard)  vs float64: "
                      f"sharded {e_got:.2e} default {e_ref:.2e}", flush=True)
                ttnn.deallocate(xs)
                break
            except Exception as exc:                                  # noqa: BLE001
                msg = (str(exc) or repr(exc)).splitlines()[0][:60]
                if sub == 1:
                    print(f"RESULT   {n_cores:3d} cores blk_w {bw:3d}: {msg}",
                          flush=True)
    ttnn.deallocate(x)

ttnn.close_mesh_device(mesh)
