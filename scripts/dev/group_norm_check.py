import sys, time, torch, ttnn
sys.path.insert(0, "scripts/dev")
from _device_model import open_model
import ttrunner_qwen38_flash_next.tt.ops as ops
mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
H, HC = cfg.hidden_size, cfg.hc_count
W = HC * H
eps = cfg.rms_norm_eps
try:
    x = ttnn.from_torch(torch.randn(1,1,1,W)*0.4, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    wt = ttnn.from_torch(torch.randn(1,1,1,W)*0.5+1.0, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    got = ops.fused_group_norm(x, wt, eps, H, HC, key=("chk",))
    if got is None: raise SystemExit("RESULT declined")
    normed_k, local_k = got
    ttnn.synchronize_device(mesh)
    ref = ops.grouped_rms_norm(x, wt, eps, H, HC)
    ref_local = ttnn.mesh_partition(ref, dim=-1)
    ttnn.synchronize_device(mesh)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    a = ttnn.to_torch(normed_k, mesh_composer=comp)[:1].to(torch.float64)
    b = ttnn.to_torch(ref, mesh_composer=comp)[:1].to(torch.float64)
    xt = ttnn.to_torch(x, mesh_composer=comp)[:1].to(torch.float64).reshape(HC, H)
    wtt = ttnn.to_torch(wt, mesh_composer=comp)[:1].to(torch.float64).reshape(HC, H)
    truth = (xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + eps) * wtt).reshape(1,1,1,W)
    sc = max(truth.abs().max().item(), 1e-30)
    print(f"RESULT normed vs float64: kernel {(a-truth).abs().max().item()/sc:.3e}, "
          f"ops {(b-truth).abs().max().item()/sc:.3e}", flush=True)
    la = ttnn.to_torch(local_k, mesh_composer=comp).to(torch.float64)
    lb = ttnn.to_torch(ref_local, mesh_composer=comp).to(torch.float64)
    print(f"RESULT local vs mesh_partition: max abs {(la-lb).abs().max().item():.3e} "
          f"shape {tuple(la.shape)}", flush=True)
    def timed(fn, label, calls=97, reps=25, iters=8):
        for _ in range(2): fn()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(reps): fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        best = float("inf")
        for _ in range(iters):
            t0 = time.perf_counter(); ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            best = min(best, time.perf_counter()-t0)
        ttnn.release_trace(mesh, tid)
        us = best/reps*1e6
        print(f"RESULT {label:36s} {us:8.2f} us x{calls} = {calls*us/1000:5.2f} ms", flush=True)
    timed(lambda: ttnn.mesh_partition(ops.grouped_rms_norm(x, wt, eps, H, HC), dim=-1),
          "grouped_rms_norm + mesh_partition")
    timed(lambda: ops.fused_group_norm(x, wt, eps, H, HC, key=("t",)), "two fused launches")
    scale = ops._GNORM_OUT[(("chk",), id(mesh), tuple(x.shape), str(x.dtype), HC)]
    part, scale, out, local = scale
    devid = ops._gnorm_devid(mesh)
    import struct
    bits = lambda f: struct.unpack("<I", struct.pack("<f", float(f)))[0]
    pa, pa2, pb = ops._gnorm_programs(x, wt, part, scale, out, local, devid, H // 32,
                                      HC, bits(1.0 / H), bits(eps), ops._GNORM_PARTS)
    timed(lambda: ttnn.generic_op([x, part], pa), "  pass 1 (partial squares)")
    timed(lambda: ttnn.generic_op([part, scale], pa2), "  pass 2 (fold)")
    timed(lambda: ttnn.generic_op([x, wt, scale, out, local, devid], pb),
          "  pass 3 (scale + weight + local)")
    # local against the kernel's own normed
    aa = ttnn.to_torch(out, mesh_composer=comp)[:1].to(torch.float64).reshape(HC, H)
    ll = ttnn.to_torch(local, mesh_composer=comp).to(torch.float64).reshape(HC, H)
    d = max((ll[i] - aa[i]).abs().max().item() for i in range(HC))
    print(f"RESULT local vs the kernel's own normed group: max abs {d:.3e}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
