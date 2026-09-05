"""The fused causal-conv kernel against the ops it replaces.

    uv run python scripts/dev/conv_kernel_check.py

Checks the output *and* the ring: the kernel advances the history in place, so a
correct column with a stale ring would pass a naive comparison and then diverge
on the next token. Both are compared against float64 on the same inputs, because
"differs from the ops" is not the same as "less accurate" -- the kernel keeps its
four taps in fp32 where the op chain rounds to bfloat16 between every step.
"""
import sys
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

C = m.conv_dim_local
K, DEPTH = cfg.conv_kernel, cfg.conv_kernel - 1


def dev(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=mesh, mesh_mapper=rep)


def host(t):
    return ttnn.to_torch(t, mesh_composer=comp)[:1].to(torch.float64)


try:
    x_t = torch.randn(1, 1, 1, C) * 0.5
    st_t = [torch.randn(1, 1, 1, C) * 0.5 for _ in range(DEPTH)]
    w_t = [torch.randn(1, 1, 1, C) * 0.5 for _ in range(K)]

    x = dev(x_t)
    taps = [dev(w) for w in w_t]

    def run(fused: bool):
        state = [dev(t) for t in st_t]
        if fused:
            out = ops.fused_conv_step(x, state, taps, key=("check", int(fused)))
            if out is None:
                raise SystemExit("RESULT fused conv declined the shape")
        else:
            acc = None
            for tap in range(K):
                age = DEPTH - tap
                piece = x if age == 0 else state[age - 1]
                term = ttnn.multiply(piece, taps[tap])
                acc = term if acc is None else ttnn.add(acc, term)
            for i in range(DEPTH - 1, 0, -1):
                ttnn.copy(state[i - 1], state[i])
            ttnn.copy(x, state[0])
            out = ttnn.silu(acc)
        ttnn.synchronize_device(mesh)
        return host(out), [host(t) for t in state]

    got, got_state = run(True)
    ref, ref_state = run(False)

    # float64 on the *rounded* operands, which is what both paths actually saw.
    xd, std, wd = host(x), [host(dev(t)) for t in st_t], [host(t) for t in taps]
    acc = wd[3] * xd + wd[2] * std[0] + wd[1] * std[1] + wd[0] * std[2]
    truth = acc * torch.sigmoid(acc)
    scale = max(truth.abs().max().item(), 1e-30)
    e_k = (got - truth).abs().max().item() / scale
    e_o = (ref - truth).abs().max().item() / scale
    print(f"RESULT output vs float64: kernel {e_k:.3e}, ops {e_o:.3e} -> "
          f"{'as accurate or better' if e_k <= e_o * 1.5 else 'WORSE'}", flush=True)
    print(f"RESULT output kernel vs ops: {(got - ref).abs().max().item():.3e}",
          flush=True)

    want = [xd, std[0], std[1]]                    # newest first, after the shift
    for i, (g, w) in enumerate(zip(got_state, want)):
        d = (g - w).abs().max().item()
        print(f"RESULT ring[{i}] after the step: max abs {d:.3e} "
              f"{'EXACT' if d == 0.0 else 'MISMATCH'}", flush=True)
    for i, (g, r) in enumerate(zip(got_state, ref_state)):
        d = (g - r).abs().max().item()
        print(f"RESULT ring[{i}] kernel vs ops:  max abs {d:.3e} "
              f"{'EXACT' if d == 0.0 else 'MISMATCH'}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)

# --- and what each costs, in a trace of its own -------------------------------
# Isolation, so invariant 77 applies: this prices the two paths against each
# other, not the change in the model. It is here to answer one question -- is the
# kernel itself slow, or were the eleven ops cheaper than 5.33 us apiece?
import time                                                        # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
try:
    C = m.conv_dim_local
    K, DEPTH = cfg.conv_kernel, cfg.conv_kernel - 1
    def dv(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=mesh, mesh_mapper=rep)
    x = dv(torch.randn(1, 1, 1, C) * 0.5)
    state = [dv(torch.randn(1, 1, 1, C) * 0.5) for _ in range(DEPTH)]
    taps = [dv(torch.randn(1, 1, 1, C) * 0.5) for _ in range(K)]

    def fused():
        return ops.fused_conv_step(x, state, taps, key=("time",))

    def chain():
        acc = None
        for tap in range(K):
            age = DEPTH - tap
            piece = x if age == 0 else state[age - 1]
            term = ttnn.multiply(piece, taps[tap])
            acc = term if acc is None else ttnn.add(acc, term)
        for i in range(DEPTH - 1, 0, -1):
            ttnn.copy(state[i - 1], state[i])
        ttnn.copy(x, state[0])
        return ttnn.silu(acc)

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
        print(f"RESULT {label:28s} {us:8.2f} us   x36 layers {36 * us / 1000:6.2f} ms",
              flush=True)
        return us
    a = timed(chain, "eleven ops")
    b = timed(fused, "one fused launch")
    print(f"RESULT -> {a / b:.2f}x, {36 * (a - b) / 1000:+.2f} ms a token", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
