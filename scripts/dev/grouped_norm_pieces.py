"""Where do `grouped_rms_norm`'s 38 us go, and what would folding save?

    uv run python scripts/dev/grouped_norm_pieces.py

It is 3.83 ms of the step across 100 calls -- four ops: a reshape that folds the
hc groups into rows, an `rms_norm`, a reshape back, and the weight multiply. Two
of those are re-tilings, and `elementwise_shape_cost.py` puts a big one at
19.8 us, so the cost may be almost entirely layout.

If it is, then carrying the hyper-connection stream in the folded shape --
[1, 1, hc*M, hidden] rather than [1, 1, M, hc*hidden] -- removes both reshapes
*and* makes the stream four times smaller in tiles at M=1 (80 against 320),
which every wide op on it then inherits. This prices that before it is built.
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
H, HC = cfg.hidden_size, cfg.hc_count
W = H * HC


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


for M in (1, 32):
    x = ttnn.from_torch(torch.randn(1, 1, M, W) * 0.5, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    xf = ttnn.from_torch(torch.randn(1, 1, HC * M, H) * 0.5, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    w = ttnn.from_torch(torch.randn(1, 1, 1, W) * 0.1, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    wf = ttnn.from_torch(torch.randn(1, 1, HC * M, H) * 0.1, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    tiles_un = max(1, (M + 31) // 32) * (W // 32)
    tiles_f = max(1, (HC * M + 31) // 32) * (H // 32)
    print(f"RESULT M={M}: unfolded [1,1,{M},{W}] is {tiles_un} tiles, "
          f"folded [1,1,{HC * M},{H}] is {tiles_f}", flush=True)

    t_r1 = timed(lambda x=x: ttnn.reshape(x, (1, 1, M * HC, H)))
    folded = ttnn.reshape(x, (1, 1, M * HC, H))
    t_n = timed(lambda f=folded: ttnn.rms_norm(f, epsilon=1e-6))
    normed = ttnn.rms_norm(folded, epsilon=1e-6)
    t_r2 = timed(lambda n=normed: ttnn.reshape(n, (1, 1, M, W)))
    back = ttnn.reshape(normed, (1, 1, M, W))
    t_m = timed(lambda b=back, w=w: ttnn.multiply(b, w))
    t_mf = timed(lambda n=normed, wf=wf: ttnn.multiply(n, wf))
    t_part_un = timed(lambda b=back: ttnn.mesh_partition(b, dim=-1))
    try:
        t_part_f = timed(lambda n=normed: ttnn.mesh_partition(n, dim=-2))
    except Exception as exc:                                          # noqa: BLE001
        # A row-axis partition slices dim -2, and a slice there must start and
        # end on a tile boundary. Four groups over four devices is one row each,
        # so the folded layout cannot feed the down projection at all.
        t_part_f = float("nan")
        print(f"RESULT   mesh_partition(dim=-2) rejected: "
              f"{(str(exc) or repr(exc)).splitlines()[-1][:100]}", flush=True)
    print(f"RESULT   reshape in {t_r1:6.2f}us  rms_norm {t_n:6.2f}  "
          f"reshape back {t_r2:6.2f}  multiply(unfolded) {t_m:6.2f}", flush=True)
    print(f"RESULT   -> now {t_r1 + t_n + t_r2 + t_m:6.2f}us   "
          f"folded {t_n + t_mf:6.2f}us (rms_norm + multiply on {tiles_f} tiles)",
          flush=True)
    print(f"RESULT   mesh_partition: unfolded {t_part_un:6.2f}us, "
          f"folded {t_part_f:6.2f}us", flush=True)
    for t in (x, xf, w, wf):
        ttnn.deallocate(t)

ttnn.close_mesh_device(mesh)
