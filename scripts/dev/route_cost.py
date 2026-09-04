"""What does MoE routing cost, op by op, at the real shapes?

moe_block measures 0.879 ms a layer with the wide expert path in place, and the
gathers and matmuls account for ~0.37 of that. The other ~0.51 -- about 24 ms a
token -- is the router chain, and this prices its pieces.
"""
import sys, time
import torch, ttnn
sys.path.insert(0, "/home/twjang/twtest/scripts/dev")
from _device_model import open_model

E, EL, K, TOPK = 512, 128, 2560, 10
mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)

def dev(t, dt=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

def timed(fn, label, reps=50, iters=8):
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
    print(f"RESULT {label:44s} {us:8.2f} us -> {48*us/1e3:6.2f} ms/48", flush=True)
    return us

x = dev(torch.randn(1, 1, 1, K) * 0.05)
rw = dev(torch.randn(1, 1, K, E) * 0.02, ttnn.float32)
probs = dev(torch.softmax(torch.randn(1, 1, 1, E), dim=-1))
local = dev(torch.softmax(torch.randn(1, 1, 1, EL), dim=-1))

t = 0
t += timed(lambda: ttnn.linear(x, rw), "router linear [2560,512] fp32")
t += timed(lambda: ttnn.softmax(probs, dim=-1), "softmax over 512")
t += timed(lambda: ttnn.topk(probs, k=TOPK, dim=-1, largest=True, sorted=True),
           f"topk k={TOPK} of {E} (global)")
t += timed(lambda: ttnn.topk(local, k=TOPK, dim=-1, largest=True, sorted=True),
           f"topk k={TOPK} of {EL} (local, added by the wide path)")
t += timed(lambda: ttnn.ge(probs, probs, dtype=ttnn.bfloat16), "ge over 512")
t += timed(lambda: ttnn.divide(probs, ttnn.sum(probs, dim=-1, keepdim=True)),
           "divide by sum over 512")
t += timed(lambda: ttnn.mesh_partition(probs, dim=-1), "mesh_partition 512 -> 128")
print("RESULT ---", flush=True)
print(f"RESULT those seven alone: {t:.1f} us a layer -> {48*t/1e3:.2f} ms a token",
      flush=True)
print(f"RESULT moe_block measures 879 us a layer, of which ~370 is the expert path",
      flush=True)
ttnn.close_mesh_device(mesh)
