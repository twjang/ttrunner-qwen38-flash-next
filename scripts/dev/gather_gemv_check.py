"""The gathering GEMV against gather + matmul, on the real expert weights.

    uv run python scripts/dev/gather_gemv_check.py

The gather is the expensive half of the wide expert path -- 40.11 us to copy
gate|up against 20.80 to multiply it -- because every selected expert's weights
cross DRAM three times. This checks that reading them through the index gives the
same answer, and sweeps how many output columns a core should own: the activation
row is re-read once per core, so fewer cores move fewer bytes, and too few cannot
pull enough DRAM between them.
"""
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device_model import open_model                                # noqa: E402

import ttrunner_qwen38_flash_next.tt.moe as moe                     # noqa: E402
import ttrunner_qwen38_flash_next.tt.ops as ops                     # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
torch.manual_seed(0)

K_SEL = moe.WIDE_EXPERTS


def timed(fn, label, calls=48, reps=20, iters=8):
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
    print(f"RESULT {label:40s} {us:8.2f} us  x{calls} = {calls * us / 1000:6.2f} ms",
          flush=True)
    return us


try:
    gw = m.w.blk(0, "ffn_gateup_exps.weight")
    dw = m.w.blk(0, "ffn_down_exps.weight")
    E, K, N = gw.shape[1], gw.shape[2], gw.shape[3]
    n_half = N // 2
    dn, dh = dw.shape[2], dw.shape[3]
    print(f"RESULT gate|up {tuple(gw.shape)} {gw.dtype}, down {tuple(dw.shape)} "
          f"{dw.dtype}, k_sel {K_SEL}", flush=True)

    ids = [(i * 37) % E for i in range(K_SEL)]
    host_idx = torch.zeros(1, 1, 1, 128, dtype=torch.int32)
    for i, e in enumerate(ids):
        host_idx[0, 0, 0, i] = e
    idx = ttnn.from_torch(host_idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=mesh, mesh_mapper=rep)

    x = ttnn.from_torch(torch.randn(1, 1, 1, K) * 0.05, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    h = ttnn.from_torch(torch.randn(1, 1, 1, K_SEL * dn) * 0.05, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

    gu_shape = (1, 1, K, K_SEL * N)
    dw_shape = (1, 1, K_SEL * dn, dh)

    def ref_gu():
        gu = moe._gather(gw, idx, K_SEL, 2, gu_shape, 0)
        return ops.fast_linear(x, gu, compute_kernel_config=ops.HIFI4)

    def ref_dn():
        d = moe._gather(dw, idx, K_SEL, 0, dw_shape, 0)
        return ops.fast_linear(h, d, compute_kernel_config=ops.HIFI4)

    for tag, ref, weights, src, mode, oshape in (
            ("gate|up", ref_gu, gw, x, 2, (1, 1, 1, K_SEL * N)),
            ("down", ref_dn, dw, h, 0, (1, 1, 1, dh))):
        got = moe.gather_gemv(src, weights, idx, K_SEL, mode, oshape, 0)
        if got is None:
            print(f"RESULT {tag}: gather gemv declined", flush=True)
            continue
        ttnn.synchronize_device(mesh)
        g = ttnn.to_torch(got, mesh_composer=comp)[:1].to(torch.float64)
        r = ttnn.to_torch(ref(), mesh_composer=comp)[:1].to(torch.float64)
        sc = max(r.abs().max().item(), 1e-30)
        # Against float64 on the *gathered* weights as ttnn dequantises them,
        # not against each other: "differs from gather+matmul" is not the same
        # as "less accurate" when the weights are bfloat4.
        gath = moe._gather(weights, idx, K_SEL, 2 if mode == 2 else 0,
                           gu_shape if mode == 2 else dw_shape, 0)
        ttnn.synchronize_device(mesh)
        wt = ttnn.to_torch(gath, mesh_composer=comp)[:1].to(torch.float64)
        xt = ttnn.to_torch(src, mesh_composer=comp)[:1].to(torch.float64)
        truth = (xt.reshape(1, -1) @ wt.reshape(wt.shape[-2], wt.shape[-1])).reshape(1, 1, 1, -1)
        ts = max(truth.abs().max().item(), 1e-30)
        print(f"RESULT {tag} vs float64: gemv {(g - truth).abs().max().item() / ts:.3e}, "
              f"gather+matmul {(r - truth).abs().max().item() / ts:.3e}", flush=True)
        print(f"RESULT {tag} gemv vs gather+matmul: max rel "
              f"{(g - r).abs().max().item() / sc:.3e}", flush=True)

        base = timed(ref, f"  {tag}: gather + matmul")
        for cols in (2, 4, 8, 16):
            us = timed(lambda w=weights, s=src, mo=mode, o=oshape, c=cols:
                       moe.gather_gemv(s, w, idx, K_SEL, mo, o, 0, c),
                       f"  {tag}: gemv, {cols} cols a core")
            print(f"RESULT   -> {base / us:.2f}x, {48 * (base - us) / 1000:+.2f} ms",
                  flush=True)
finally:
    ttnn.close_mesh_device(mesh)
