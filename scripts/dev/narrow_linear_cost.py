"""The remaining narrow-output linears, and what fusing each pair would save."""
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
    print(f"RESULT {label:44s} {us:7.2f} us x{calls:3d} -> {calls*us/1e3:6.2f} ms", flush=True)
    return calls * us / 1e3

x = dev(torch.randn(1, 1, 1, 2560) * 0.05)
w48 = dev(torch.randn(1, 1, 2560, 48) * 0.02, ttnn.float32)
w96 = dev(torch.randn(1, 1, 2560, 96) * 0.02, ttnn.float32)
w640 = dev(torch.randn(1, 1, 2560, 640) * 0.02, ttnn.bfloat8_b)
w1280 = dev(torch.randn(1, 1, 2560, 1280) * 0.02, ttnn.bfloat8_b)

a = timed(lambda: ttnn.linear(x, w48), "ssm_alpha [2560,48] fp32", 36)
a += timed(lambda: ttnn.linear(x, w48), "ssm_beta  [2560,48] fp32", 36)
b = timed(lambda: ttnn.linear(x, w96), "fused alpha|beta [2560,96]", 36)
print(f"RESULT  -> fusing alpha|beta saves {a-b:+.2f} ms a token", flush=True)

c = timed(lambda: ttnn.linear(x, w640), "ffn_gate_shexp [2560,640]", 48)
c += timed(lambda: ttnn.linear(x, w640), "ffn_up_shexp   [2560,640]", 48)
d = timed(lambda: ttnn.linear(x, w1280), "fused shexp gate|up [2560,1280]", 48)
print(f"RESULT  -> fusing shexp gate|up saves {c-d:+.2f} ms a token", flush=True)
print(f"RESULT total available here: {(a-b)+(c-d):+.2f} ms a token", flush=True)
ttnn.close_mesh_device(mesh)
