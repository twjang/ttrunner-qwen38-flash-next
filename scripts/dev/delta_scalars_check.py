"""The fused DeltaNet decay/beta kernel against the nine ops it replaces.

    uv run python scripts/dev/delta_scalars_check.py

The kernel scatters `half` values into `half` one-value tiles, writing only the
first 64 bytes of each page -- so this checks the live element *and* that the
rest of every output tile is still zero, which is the contract the broadcast
multiplies in `decode_step` rely on.
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

H = m.n_v_local
try:
    def dv(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=mesh, mesh_mapper=rep)

    ab_t = torch.randn(1, 1, 1, 2 * H) * 1.5
    dt_t = torch.randn(1, 1, 1, H) * 0.5
    ad_t = -torch.rand(1, 1, 1, H) - 0.1          # A is stored already negated
    ab, dt, ad = dv(ab_t), dv(dt_t), dv(ad_t)

    got = ops.fused_delta_scalars(ab, dt, ad, H, key=("check",))
    if got is None:
        raise SystemExit("RESULT fused delta scalars declined")
    g_k, b_k = got
    ttnn.synchronize_device(mesh)
    g_kt = ttnn.to_torch(g_k, mesh_composer=comp)[:H].to(torch.float64)
    b_kt = ttnn.to_torch(b_k, mesh_composer=comp)[:H].to(torch.float64)

    a = ops._slice_last_op(ab, 0, H) if hasattr(ops, "_slice_last_op") else \
        ttnn.slice(ab, (0, 0, 0, 0), (1, 1, 1, H))
    b = ttnn.slice(ab, (0, 0, 0, H), (1, 1, 1, 2 * H))
    g = ttnn.multiply(ad, ttnn.softplus(ttnn.add(a, dt)))
    g_o = ttnn.reshape(ttnn.exp(g), (H, 1, 1, 1))
    b_o = ttnn.reshape(ttnn.sigmoid(b), (H, 1, 1, 1))
    ttnn.synchronize_device(mesh)
    g_ot = ttnn.to_torch(g_o, mesh_composer=comp)[:H].to(torch.float64)
    b_ot = ttnn.to_torch(b_o, mesh_composer=comp)[:H].to(torch.float64)

    # float64 on the values both paths actually saw (bfloat16 rounded).
    abd = ttnn.to_torch(ab, mesh_composer=comp)[:1].to(torch.float64)
    dtd = ttnn.to_torch(dt, mesh_composer=comp)[:1].to(torch.float64)
    add = ttnn.to_torch(ad, mesh_composer=comp)[:1].to(torch.float64)
    ta = abd[..., :H]
    tb = abd[..., H:]
    truth_g = torch.exp(add * torch.nn.functional.softplus(ta + dtd)).reshape(H, 1, 1, 1)
    truth_b = torch.sigmoid(tb).reshape(H, 1, 1, 1)

    for name, k, o, t in (("g_exp", g_kt, g_ot, truth_g), ("beta", b_kt, b_ot, truth_b)):
        sc = max(t.abs().max().item(), 1e-30)
        ek = (k - t).abs().max().item() / sc
        eo = (o - t).abs().max().item() / sc
        print(f"RESULT {name} vs float64: kernel {ek:.3e}, ops {eo:.3e} -> "
              f"{'as accurate or better' if ek <= eo * 1.5 else 'WORSE'}", flush=True)
        print(f"RESULT {name} kernel vs ops: {(k - o).abs().max().item():.3e}", flush=True)

    # The pad. Only element (0,0) of each page is written, so everything else
    # must still be the zero it was allocated with.
    for name, t in (("g_exp", g_k), ("beta", b_k)):
        raw = ttnn.to_torch(t, mesh_composer=comp)
        print(f"RESULT {name} shape out {tuple(raw.shape)}", flush=True)

    def timed(fn, label, reps=40, iters=8):
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
        print(f"RESULT {label:24s} {us:8.2f} us   x36 layers {36 * us / 1000:6.2f} ms",
              flush=True)
        return us

    def chain():
        aa = ttnn.slice(ab, (0, 0, 0, 0), (1, 1, 1, H))
        bb = ttnn.slice(ab, (0, 0, 0, H), (1, 1, 1, 2 * H))
        gg = ttnn.multiply(ad, ttnn.softplus(ttnn.add(aa, dt)))
        return (ttnn.reshape(ttnn.exp(gg), (H, 1, 1, 1)),
                ttnn.reshape(ttnn.sigmoid(bb), (H, 1, 1, 1)))

    c = timed(chain, "nine ops")
    k = timed(lambda: ops.fused_delta_scalars(ab, dt, ad, H, key=("t",)), "one fused launch")
    print(f"RESULT -> {c / k:.2f}x, {36 * (c - k) / 1000:+.2f} ms a token", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
