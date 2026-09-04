"""Why does a GEMV get nowhere near DRAM bandwidth?

    uv run python scripts/dev/gemv_bandwidth_check.py

A GEMV should be bandwidth-bound by construction: hold x and b in SRAM, stream
the weight once, accumulate. Whatever bandwidth the device has, that is the
speed. Measured, a decode projection reads its weight at 73.6 GB/s, which is far
short of what this hardware should do -- so either the ceiling is lower than
expected or the op is not reaching it.

The tell is already in the data: the same projection takes 0.060 ms with a
bfloat16 weight (15.7 MB) *and* with bfloat4_b (4.42 MB). Same time for 3.6x
less data is not a bandwidth limit; it is a floor.

This sweeps the weight's size and dtype to find where that floor sits and where
bandwidth actually takes over, and measures a plain elementwise op for the
device's achievable streaming bandwidth to compare against. Timed as throughput
(a run of calls, one synchronise), because per-call timing measures dispatch.
"""
import time

import torch
import ttnn

from ttrunner_qwen38_flash_next.tt.ops import HIFI4

K = 2560
NS = (256, 1024, 3072, 8192, 16384, 32768)
BATCH = 30

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)


def throughput(fn, batch=BATCH):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(mesh)
    best = None
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(batch):
            fn()
        ttnn.synchronize_device(mesh)
        dt = (time.perf_counter() - t0) / batch
        best = dt if best is None else min(best, dt)
    return best


# --- what the device can stream at all -------------------------------------
big = ttnn.from_torch(torch.randn(1, 1, 4096, 4096), dtype=ttnn.bfloat16,
                      layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
mb = 4096 * 4096 * 2 / 1e6
s = throughput(lambda: ttnn.multiply(big, 2.0), batch=10)
print(f"RESULT elementwise x*2 on {mb:.0f} MB: {1000 * s:.3f} ms -> "
      f"{2 * mb / 1e3 / s:.0f} GB/s (read+write)", flush=True)
ttnn.deallocate(big)

# --- the GEMV, swept ---------------------------------------------------------
x = ttnn.from_torch(torch.randn(1, 1, 32, K) * 0.1, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
print("RESULT  dtype        N   weight MB    time      GB/s", flush=True)
for dtype, per in ((ttnn.bfloat4_b, 0.5625), (ttnn.bfloat16, 2.0)):
    for N in NS:
        w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.05, dtype=dtype,
                            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        wmb = K * N * per / 1e6
        s = throughput(lambda: ttnn.linear(x, w, compute_kernel_config=HIFI4))
        name = "bfloat4_b" if per < 1 else "bfloat16 "
        print(f"RESULT  {name} {N:6d}   {wmb:8.2f}   {1000 * s:7.3f} ms  {wmb / 1e3 / s:7.1f}",
              flush=True)
        ttnn.deallocate(w)

ttnn.close_mesh_device(mesh)
