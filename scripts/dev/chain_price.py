"""What each remaining fusible chain costs, in a trace of its own.

    uv run python scripts/dev/chain_price.py

Invariant 77 says to price a change by inserting it into the model, and that
still holds for deciding whether a *built* kernel pays. This is the step before
that: which chains are even worth building a kernel for. A chain that costs 8 us
cannot return 1 ms over 36 layers however well it is fused.

Each entry is the ops as the model issues them, on tensors of the model's shapes,
with the per-token total at the call count the census measured.
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
torch.manual_seed(0)


def dv(*shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(*shape) * 0.3, dtype=dtype,
                           layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)


def timed(fn, label, calls, reps=30, iters=8):
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
    print(f"RESULT {label:34s} {us:7.2f} us  x{calls:4d} = {calls * us / 1000:5.2f} ms",
          flush=True)


try:
    H, HC = cfg.hidden_size, cfg.hc_count
    W = HC * H
    hd, n_v = cfg.linear_head_dim, m.n_v_local
    V = n_v * hd

    # -- the hyper-connection norm, six ops, 100 calls -------------------------
    x, wgt = dv(1, 1, 1, W), dv(1, 1, 1, W)
    timed(lambda: ops.grouped_rms_norm(x, wgt, 1e-6, H, HC), "grouped_rms_norm (6 ops)", 100)

    # -- the DeltaNet tail, six ops, 36 calls ----------------------------------
    out_h, z = dv(n_v, 1, 1, hd), dv(1, 1, 1, V)
    nw = dv(1, 1, 1, hd)

    def tail():
        o = ttnn.reshape(out_h, (1, 1, n_v, hd))
        zh = ttnn.reshape(z, (1, 1, n_v, hd))
        n = ttnn.rms_norm(o, epsilon=1e-6, weight=nw, compute_kernel_config=ops.HIFI4)
        g = ttnn.multiply(n, ttnn.sigmoid(zh))
        return ttnn.reshape(g, (1, 1, 1, V))
    timed(tail, "DeltaNet tail (6 ops)", 36)

    # -- q/k l2norm, four ops, 36 calls ----------------------------------------
    q, kk = dv(n_v, 1, 1, hd), dv(n_v, 1, 1, hd)
    timed(lambda: (m._l2norm(q, scale=hd ** -0.5), m._l2norm(kk)), "q/k l2norm (4 ops)", 36)

    # -- decode_step's non-matmul interior, 36 calls ---------------------------
    st = dv(n_v, 1, hd, hd, dtype=ttnn.float32)
    ge, be = dv(n_v, 1, 1, 1), dv(n_v, 1, 1, 1)
    v = dv(n_v, 1, 1, hd)

    def dstep():
        dec = ttnn.multiply(st, ge)
        kq = ttnn.concat([kk, q], dim=-2)
        both = ttnn.matmul(kq, dec, compute_kernel_config=ops.HIFI4)
        s = list(both.shape)
        pred = ttnn.slice(both, (0, 0, 0, 0), (s[0], s[1], 1, s[3]))
        qd = ttnn.slice(both, (0, 0, 1, 0), (s[0], s[1], 2, s[3]))
        delta = ttnn.multiply(ttnn.subtract(v, pred), be)
        upd = ttnn.matmul(ttnn.transpose(kk, -2, -1), delta, compute_kernel_config=ops.HIFI4)
        ttnn.add(dec, upd, output_tensor=st)
        qk = ttnn.sum(ttnn.multiply(q, kk), dim=-1, keepdim=True)
        return ttnn.add(qd, ttnn.multiply(qk, delta))
    timed(dstep, "decode_step (11 ops, 2 matmuls)", 36)

    # -- decode_step, op by op -------------------------------------------------
    dec0 = ttnn.multiply(st, ge)
    kq0 = ttnn.concat([kk, q], dim=-2)
    both0 = ttnn.matmul(kq0, dec0, compute_kernel_config=ops.HIFI4)
    s0 = list(both0.shape)
    pred0 = ttnn.slice(both0, (0, 0, 0, 0), (s0[0], s0[1], 1, s0[3]))
    delta0 = ttnn.multiply(ttnn.subtract(v, pred0), be)
    kt0 = ttnn.transpose(kk, -2, -1)
    upd0 = ttnn.matmul(kt0, delta0, compute_kernel_config=ops.HIFI4)
    print("RESULT --- decode_step, piece by piece ---", flush=True)
    timed(lambda: ttnn.multiply(st, ge), "  multiply(state, g_exp) 192t f32", 36)
    timed(lambda: ttnn.concat([kk, q], dim=-2), "  concat([k, q])", 36)
    timed(lambda: ttnn.matmul(kq0, dec0, compute_kernel_config=ops.HIFI4),
          "  matmul [12,1,2,128]x[..,128,128]", 36)
    timed(lambda: ttnn.slice(both0, (0, 0, 0, 0), (s0[0], s0[1], 1, s0[3])), "  slice", 72)
    timed(lambda: ttnn.multiply(ttnn.subtract(v, pred0), be), "  subtract + multiply", 36)
    timed(lambda: ttnn.transpose(kk, -2, -1), "  transpose(k)", 36)
    timed(lambda: ttnn.matmul(kt0, delta0, compute_kernel_config=ops.HIFI4),
          "  matmul [12,1,128,1]x[..,1,128]", 36)
    timed(lambda: ttnn.add(dec0, upd0, output_tensor=st), "  add -> state (in place)", 36)
    timed(lambda: ttnn.sum(ttnn.multiply(q, kk), dim=-1, keepdim=True), "  qk = sum(q*k)", 36)

    # -- the k-split reduction, 193 calls --------------------------------------
    parts = dv(1, 11, 32, 320)
    timed(lambda: ttnn.sum(parts, dim=1, keepdim=True), "ksplit sum (1 op)", 193)
finally:
    ttnn.close_mesh_device(mesh)
