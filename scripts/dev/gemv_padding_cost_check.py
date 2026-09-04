"""What does the 32x row padding actually cost a GEMV?

    uv run python scripts/dev/gemv_padding_cost_check.py

The intuition to test: at M=1 the activation is padded to 32 rows, so the machine
does 32x the arithmetic it needs and a decode GEMV runs at 1/32 of its potential
speed. If that is right, removing the padding is worth up to 32x and is the
single biggest thing on the table.

It is decidable without writing a new kernel. Sweep M against a fixed weight:

* If the padding is what costs, M=1 and M=32 take the *same* time -- both compute
  32 rows -- and time rises from M=32 on. The 31 wasted rows are then real
  wasted capacity, and a 1-row kernel could in principle reclaim them.
* If instead the op is bound by fetching the weight, M=1 and M=32 also take the
  same time... which is why the sweep has to go further. What separates the two
  is the *slope* past 32 and the achieved bandwidth: a weight-bound op keeps
  taking the same time well past M=32, because the weight is read once whatever
  M is.

Timed as throughput (a run of calls, one synchronise) because a per-call timing
measures dispatch, not the op (invariant 23).
"""
import time

import torch
import ttnn

from ttrunner_qwen38_flash_next.tt.ops import HIFI4

K, N = 2560, 3072                 # QSA q|gate, per device
MS = (1, 2, 8, 16, 32, 64, 128, 256, 512)
BYTES_PER_W = 0.5625              # bfloat4_b: 4 bits + a shared exponent per 16
BATCH = 40

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.05, dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
w_mb = K * N * BYTES_PER_W / 1e6

print(f"RESULT weight {K}x{N} bfloat4_b = {w_mb:.2f} MB per device", flush=True)
print("RESULT    M   time/call   rows/ms   weight GB/s   vs M=1", flush=True)
base = None
for m in MS:
    x = ttnn.from_torch(torch.randn(1, 1, m, K) * 0.1, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

    def fn(x=x):
        return ttnn.linear(x, w, compute_kernel_config=HIFI4)

    for _ in range(3):
        fn()
    ttnn.synchronize_device(mesh)
    best = None
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(BATCH):
            fn()
        ttnn.synchronize_device(mesh)
        dt = (time.perf_counter() - t0) / BATCH
        best = dt if best is None else min(best, dt)
    base = base or best
    print(f"RESULT {m:5d}   {1000 * best:7.3f} ms   {m / (1000 * best):7.1f}   "
          f"{w_mb / 1e3 / best:9.1f}   {best / base:5.2f}x", flush=True)
    ttnn.deallocate(x)

ttnn.close_mesh_device(mesh)
