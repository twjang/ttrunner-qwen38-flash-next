"""Why is the MoE's gate|up matmul at 19 % when a wider one hits 93 %?

    uv run python scripts/dev/matmul_dtype_width.py

`linear_shape_census.py` puts 2.97 ms a token on `[2560, 3200] bfloat4_b` at
19.2 % of bandwidth, and 1.11 ms on `[2560, 12288] bfloat8_b` at **93 %**. Both
are M=1 GEMVs on a 110-core grid; the second has 384 output tiles and the first
100, so neither is starved of parallelism in the way invariant 38 describes.

Two things differ: the width, and the dtype. This separates them -- the same
widths at bfloat4_b, bfloat8_b and bfloat16 -- because the answer decides what
can be done about it. If it is width, gather the experts wider. If it is the
4-bit unpack, the choice is bfloat8_b at twice the resident bytes, and that is a
memory budget question rather than a kernel one.
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

K = 2560
WIDTHS = [1280, 3200, 6400, 12288]
DTYPES = [("bfloat4_b", ttnn.bfloat4_b, 0.5625),
          ("bfloat8_b", ttnn.bfloat8_b, 1.0625),
          ("bfloat16", ttnn.bfloat16, 2.0)]


def timed(fn, reps=30, iters=8):
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


x = ttnn.from_torch(torch.randn(1, 1, 1, K) * 0.05, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
print(f"RESULT K={K}, M=1, grid "
      f"{mesh.compute_with_storage_grid_size().x}x{mesh.compute_with_storage_grid_size().y}",
      flush=True)
print(f"RESULT {'N':>6s} {'tiles':>6s} " + " ".join(f"{d[0]:>22s}" for d in DTYPES), flush=True)
for n in WIDTHS:
    cells = []
    for name, dt, bpe in DTYPES:
        try:
            w = ttnn.from_torch(torch.randn(1, 1, K, n) * 0.02, dtype=dt,
                                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            us = timed(lambda x=x, w=w: ttnn.linear(w=w, input_tensor_a=x, input_tensor_b=w)
                       if False else ttnn.linear(x, w))
            ttnn.deallocate(w)
            roof = K * n * bpe / 388e9 * 1e6
            cells.append(f"{us:8.2f}us {roof/us*100:5.1f}% {K*n*bpe/us/1e3:6.0f}GB/s")
        except Exception as exc:                                      # noqa: BLE001
            cells.append(f"{type(exc).__name__:>22s}")
    print(f"RESULT {n:6d} {n//32:6d} " + " ".join(cells), flush=True)

ttnn.close_mesh_device(mesh)
