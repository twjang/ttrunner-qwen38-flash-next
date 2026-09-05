"""The fused DeltaNet tail against the six ops it replaces."""
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device_model import open_model                                # noqa: E402

import ttrunner_qwen38_flash_next.tt.ops as ops                     # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
torch.manual_seed(0)

hd, n_v, V = cfg.linear_head_dim, m.n_v_local, m.value_dim_local
eps = cfg.rms_norm_eps
try:
    def dv(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=mesh, mesh_mapper=rep)

    o_t = torch.randn(n_v, 1, 1, hd) * 0.3
    z_t = torch.randn(1, 1, 1, V) * 0.5
    w_t = torch.randn(1, 1, 1, hd) * 0.5 + 1.0
    o, z, w = dv(o_t), dv(z_t), dv(w_t)

    got = ops.fused_delta_tail(o, z, w, n_v, hd, eps, key=("check",))
    if got is None:
        raise SystemExit("RESULT fused delta tail declined")
    ttnn.synchronize_device(mesh)
    g = ttnn.to_torch(got, mesh_composer=comp)[:1].to(torch.float64)

    def ops_path():
        oo = ttnn.reshape(o, (1, 1, n_v, hd))
        zz = ttnn.reshape(z, (1, 1, n_v, hd))
        nn = ttnn.rms_norm(oo, epsilon=eps, weight=w, compute_kernel_config=ops.HIFI4)
        gg = ttnn.multiply(nn, ttnn.sigmoid(zz))
        return ttnn.reshape(gg, (1, 1, 1, V))

    r = ttnn.to_torch(ops_path(), mesh_composer=comp)[:1].to(torch.float64)

    ro = ttnn.to_torch(o, mesh_composer=comp)[:n_v].to(torch.float64).reshape(n_v, hd)
    rz = ttnn.to_torch(z, mesh_composer=comp)[:1].to(torch.float64).reshape(n_v, hd)
    rw = ttnn.to_torch(w, mesh_composer=comp)[:1].to(torch.float64).reshape(1, hd)
    truth = (ro * torch.rsqrt(ro.pow(2).mean(-1, keepdim=True) + eps) * rw
             * torch.sigmoid(rz)).reshape(1, 1, 1, V)
    sc = max(truth.abs().max().item(), 1e-30)
    ek = (g - truth).abs().max().item() / sc
    eo = (r - truth).abs().max().item() / sc
    print(f"RESULT vs float64: kernel {ek:.3e}, ops {eo:.3e} -> "
          f"{'as accurate or better' if ek <= eo * 1.5 else 'WORSE'}", flush=True)
    print(f"RESULT kernel vs ops: {(g - r).abs().max().item():.3e}", flush=True)

    def timed(fn, label, reps=30, iters=8):
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
        print(f"RESULT {label:24s} {us:7.2f} us   x36 layers {36 * us / 1000:5.2f} ms",
              flush=True)
        return us

    a = timed(ops_path, "six ops")
    b = timed(lambda: ops.fused_delta_tail(o, z, w, n_v, hd, eps, key=("t",)),
              "one fused launch")
    print(f"RESULT -> {a / b:.2f}x, {36 * (a - b) / 1000:+.2f} ms a token", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
