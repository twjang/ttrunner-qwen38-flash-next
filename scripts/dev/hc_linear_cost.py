"""What do the hyper-connection linears cost, at their real shapes?

`ttnn.linear` is 880 calls a step and about 43 ms of the 91, and 288 of those
calls are the hc block: down, up and inject, twice a layer over 48 layers.
`hc_inject` is [10240, 4] -- four output columns, one tile -- and invariant 38
says achieved bandwidth tracks the output width. It has never been measured.
"""
import sys, time
import torch, ttnn
sys.path.insert(0, "/home/twjang/twtest/scripts/dev")
from _device_model import open_model

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)

def dev(t, dt=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

def timed(fn, label, calls, reps=50, iters=8):
    for _ in range(2): fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(reps): fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter(); ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    us = best / reps * 1e6
    print(f"RESULT {label:40s} {us:8.2f} us x{calls:4d} -> {calls*us/1e3:6.2f} ms/token",
          flush=True)
    return calls * us / 1e3

x10240 = dev(torch.randn(1, 1, 1, 10240) * 0.05)
x2560 = dev(torch.randn(1, 1, 1, 2560) * 0.05)
x320 = dev(torch.randn(1, 1, 1, 320) * 0.05)

# Allocated before any capture: `ttnn.from_torch` inside a traced region is a
# host write, which the queue refuses outright.
w_inj = dev(torch.randn(1, 1, 10240, 4) * 0.02, ttnn.float32)
w_down = dev(torch.randn(1, 1, 2560, 320) * 0.02, ttnn.bfloat8_b)
w_up = dev(torch.randn(1, 1, 320, 10240) * 0.02, ttnn.bfloat8_b)

tot = 0
tot += timed(lambda: ttnn.linear(x10240, w_inj), "hc_inject [10240, 4] fp32", 96)
tot += timed(lambda: ttnn.linear(x2560, w_down), "hc_down (K-sharded) [2560, 320]", 96)
tot += timed(lambda: ttnn.linear(x320, w_up), "hc_up [320, 10240]", 96)
print("RESULT ---", flush=True)
print(f"RESULT the hc linears together: {tot:.2f} ms a token", flush=True)
print(f"RESULT (linear is 880 calls and ~43 ms of the 91.09 ms step)", flush=True)
ttnn.close_mesh_device(mesh)
