"""The fused q/k/v head split against the ten ops it replaces.

    uv run python scripts/dev/qkv_heads_check.py
"""
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

hd, n_v = cfg.linear_head_dim, m.n_v_local
kd, vd, C = m.key_dim_local, m.value_dim_local, m.conv_dim_local
try:
    t = torch.randn(1, 1, 1, C) * 0.3
    qkv = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=mesh, mesh_mapper=rep)

    got = ops.fused_qkv_heads(qkv, kd, vd, n_v, hd, 1e-6, hd ** -0.5, 1.0,
                              key=("check",))
    if got is None:
        raise SystemExit("RESULT fused qkv heads declined")
    ttnn.synchronize_device(mesh)
    gq, gk, gv = (ttnn.to_torch(x, mesh_composer=comp)[:n_v].to(torch.float64) for x in got)

    def ops_path():
        a = m._slice_last(qkv, 0, kd)
        b = m._slice_last(qkv, kd, 2 * kd)
        c = m._slice_last(qkv, 2 * kd, 2 * kd + vd)
        a = ttnn.reshape(a, (n_v, 1, 1, hd))
        b = ttnn.reshape(b, (n_v, 1, 1, hd))
        c = ttnn.reshape(c, (n_v, 1, 1, hd))
        return m._l2norm(a, scale=hd ** -0.5), m._l2norm(b), c

    oq, ok, ov = ops_path()
    ttnn.synchronize_device(mesh)
    rq, rk, rv = (ttnn.to_torch(x, mesh_composer=comp)[:n_v].to(torch.float64)
                  for x in (oq, ok, ov))

    # float64 on the bfloat16 the device actually saw
    raw = ttnn.to_torch(qkv, mesh_composer=comp)[:1].to(torch.float64).reshape(-1)
    tq = raw[:kd].reshape(n_v, 1, 1, hd)
    tk = raw[kd:2 * kd].reshape(n_v, 1, 1, hd)
    tv = raw[2 * kd:2 * kd + vd].reshape(n_v, 1, 1, hd)
    truth_q = tq * (hd ** -0.5) / torch.sqrt(tq.pow(2).sum(-1, keepdim=True) + 1e-6)
    truth_k = tk / torch.sqrt(tk.pow(2).sum(-1, keepdim=True) + 1e-6)

    for name, g, r, truth in (("q", gq, rq, truth_q), ("k", gk, rk, truth_k),
                              ("v", gv, rv, tv)):
        sc = max(truth.abs().max().item(), 1e-30)
        ek = (g - truth).abs().max().item() / sc
        eo = (r - truth).abs().max().item() / sc
        print(f"RESULT {name} vs float64: kernel {ek:.3e}, ops {eo:.3e} -> "
              f"{'as accurate or better' if ek <= eo * 1.5 else 'WORSE'}", flush=True)
        print(f"RESULT {name} kernel vs ops: {(g - r).abs().max().item():.3e}", flush=True)

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
        print(f"RESULT {label:26s} {us:7.2f} us   x36 layers {36 * us / 1000:5.2f} ms",
              flush=True)
        return us

    a = timed(ops_path, "ten ops")
    b = timed(lambda: ops.fused_qkv_heads(qkv, kd, vd, n_v, hd, 1e-6, hd ** -0.5, 1.0,
                                          key=("t",)), "one fused launch")
    print(f"RESULT -> {a / b:.2f}x, {36 * (a - b) / 1000:+.2f} ms a token", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
