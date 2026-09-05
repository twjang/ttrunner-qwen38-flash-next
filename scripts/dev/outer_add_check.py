"""`state = decayed + kt (x) delta` in one launch, against the two ops it replaces.

    uv run python scripts/dev/outer_add_check.py

The DeltaNet state update materialises `update = kt (x) delta` -- a full
state-sized tensor, 786 KB at batch 1 -- that the following add reads once and
nothing else looks at. Thirty-six layers a token is 57 MB whose only job is to
carry a value between two launches. Accuracy against float64 first, since the
fused form rounds once where the pair rounds twice.
"""
import sys, time, torch, ttnn
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/dev"))
from _device_model import open_model                                # noqa: E402
import ttrunner_qwen38_flash_next.tt.ops as ops                     # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
torch.manual_seed(0)
try:
    BH, DK, DV = 12, 128, 128
    d_t = torch.randn(BH, 1, DK, DV) * 0.1
    k_t = torch.randn(BH, 1, DK, 1) * 0.3
    v_t = torch.randn(BH, 1, 1, DV) * 0.3
    mk = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT,
                                       device=mesh, mesh_mapper=rep)
    decayed = mk(d_t, ttnn.float32)
    kt = mk(k_t, ttnn.bfloat16)
    delta = mk(v_t, ttnn.bfloat16)
    state = mk(torch.zeros(BH, 1, DK, DV), ttnn.float32)
    ref_state = mk(torch.zeros(BH, 1, DK, DV), ttnn.float32)

    ok = ops.fused_outer_add(decayed, kt, delta, state)
    print(f"RESULT fused ran: {ok}", flush=True)
    if not ok:
        raise SystemExit("RESULT declined")
    ttnn.add(decayed, ttnn.multiply(kt, delta), output_tensor=ref_state)
    ttnn.synchronize_device(mesh)

    # float64 on the operands the device actually holds, so the comparison is
    # against the truth rather than against the pair of ops.
    d64 = ttnn.to_torch(decayed, mesh_composer=comp)[:BH].to(torch.float64)
    k64 = ttnn.to_torch(kt, mesh_composer=comp)[:BH].to(torch.float64)
    v64 = ttnn.to_torch(delta, mesh_composer=comp)[:BH].to(torch.float64)
    truth = d64 + k64 * v64
    sc = max(truth.abs().max().item(), 1e-30)
    got = ttnn.to_torch(state, mesh_composer=comp)[:BH].to(torch.float64)
    ref = ttnn.to_torch(ref_state, mesh_composer=comp)[:BH].to(torch.float64)
    r_mine = (got - truth).abs().max().item() / sc
    r_ops = (ref - truth).abs().max().item() / sc
    verdict = "as accurate" if r_mine <= r_ops * 1.5 else f"{r_mine / max(r_ops,1e-30):.1f}x worse"
    print(f"RESULT vs float64: kernel {r_mine:.3e}, ops {r_ops:.3e} -> {verdict}", flush=True)

    def timed(fn, tag, calls=36, reps=25, iters=8):
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
        print(f"RESULT {tag:28s} {us:8.2f} us x{calls} = {calls*us/1000:5.2f} ms", flush=True)
        return us

    a = timed(lambda: ttnn.add(decayed, ttnn.multiply(kt, delta), output_tensor=ref_state),
              "multiply + add")
    b = timed(lambda: ops.fused_outer_add(decayed, kt, delta, state), "fused outer add")
    print(f"RESULT -> {a/b:.2f}x, {36*(a-b)/1000:+.3f} ms a token", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
