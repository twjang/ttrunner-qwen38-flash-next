"""The k-split GEMV against `ttnn.linear`, on the shapes the census ranks worst.

    uv run python scripts/dev/ksgemv_check.py

`linear_shape_census.py` says 9.58 ms a token goes through `ttnn.linear` against
a 5.80 ms roofline, and the gap is concentrated in the narrow outputs: at M = 1
each output tile goes to one core, so `[2560, 352]` fills eleven of a hundred and
ten. Accuracy first -- against float64 on the quantised operands, not against
`ttnn.linear`, since the split changes the summation order and "differs" is not
"worse" -- then the time.
"""
import sys, time, torch, ttnn
sys.path.insert(0, "scripts/dev")
from _device_model import open_model
import ttrunner_qwen38_flash_next.tt.ops as ops

TILE = 32
SHAPES = [
    ("hc down|inject [2560, 352]", 2560, 352, ttnn.bfloat8_b),
    ("router        [2560, 512]",  2560, 512, ttnn.bfloat16),
    ("qsa small     [2560, 512]",  2560, 512, ttnn.bfloat8_b),
    ("indexer       [2560, 128]",  2560, 128, ttnn.bfloat16),
    ("ssm ab        [2560,  32]",  2560,  32, ttnn.float32),
    ("attn out      [2560, 1312]", 2560, 1312, ttnn.bfloat8_b),
    ("wide          [2560, 2560]", 2560, 2560, ttnn.bfloat8_b),
]

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
torch.manual_seed(0)
try:
    grid = mesh.compute_with_storage_grid_size()
    print(f"RESULT grid {grid.x}x{grid.y} = {grid.x * grid.y} cores", flush=True)
    for label, K, N, wdt in SHAPES:
        a = ttnn.from_torch(torch.randn(1, 1, 1, K) * 0.05, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.02, dtype=wdt,
                            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        plan = ops._ksgemv_plan(grid, K // TILE, N // TILE)
        got = ops.ksgemv(a, w, key=(label,))
        if got is None:
            print(f"RESULT {label}: declined (plan {plan})", flush=True)
            continue
        ref = ttnn.linear(a, w)
        ttnn.synchronize_device(mesh)
        a64 = ttnn.to_torch(a, mesh_composer=comp)[:1].to(torch.float64)
        w64 = ttnn.to_torch(w, mesh_composer=comp)[:1].to(torch.float64)
        truth = a64.reshape(-1, K)[:1] @ w64.reshape(K, N)
        sc = max(truth.abs().max().item(), 1e-30)
        f = lambda t: (ttnn.to_torch(t, mesh_composer=comp)[:1]
                       .to(torch.float64).reshape(-1, N)[:1] - truth).abs().max().item() / sc
        r_mine, r_ttnn = f(got), f(ref)
        verdict = ("as accurate" if r_mine <= r_ttnn * 1.5
                   else f"{r_mine / max(r_ttnn, 1e-30):.1f}x worse")
        cores_pg, groups, _ = plan
        print(f"RESULT {label}  groups={groups:3d} cores/group={cores_pg:3d}", flush=True)
        print(f"RESULT   vs float64: kernel {r_mine:.3e}, ttnn.linear {r_ttnn:.3e}"
              f" -> {verdict}", flush=True)

        def timed(fn, tag, reps=25, iters=8):
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
            print(f"RESULT   {tag:24s} {us:8.2f} us", flush=True)
            return us

        base = timed(lambda: ttnn.linear(a, w), "ttnn.linear")
        mine = timed(lambda: ops.ksgemv(a, w, key=(label,)), "k-split gemv")
        print(f"RESULT   -> {base / mine:.2f}x", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
