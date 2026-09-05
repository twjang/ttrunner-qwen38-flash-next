"""Is the M=1 matmul slow because of the program config it picks by default?

    uv run python scripts/dev/matmul_program_config.py

Three things have been ruled out. `matmul_dtype_width.py`: the time does not
depend on the weight's dtype ([2560, 3200] is 61.4 us at bfloat4_b, bfloat8_b
*and* bfloat16), so it is not bandwidth. `matmul_fidelity.py`: HiFi4, HiFi3,
HiFi2 and LoFi are all within 2 %, so it is not the FPU. What is left tracks the
length of the K loop -- roughly 0.65 us a k-tile across every shape in the
census -- which looks like read latency rather than throughput.

`ttnn.linear` with no program config picks one itself. The 1D multicast config
takes `in0_block_w`, which is how many k-tiles a core pulls before it computes:
raise it and the reads pipeline. This tries the plausible settings on the shapes
that hold the 27 ms, against the default.
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
torch.manual_seed(0)
grid = mesh.compute_with_storage_grid_size()
N_CORES = grid.x * grid.y

SHAPES = [
    ("MoE down      [1600, 2560]", 1600, 2560, ttnn.bfloat8_b),   # kt 50: no 4 or 8
    ("hc_up          [320, 10240]", 320, 10240, ttnn.bfloat8_b),  # kt 10
    ("MoE gate|up   [2560, 3200]", 2560, 3200, ttnn.bfloat4_b),   # kt 80
    ("attn_output   [6144, 2560]", 6144, 2560, ttnn.bfloat8_b),   # kt 192
    ("attn_q|gate   [2560, 12288]", 2560, 12288, ttnn.bfloat8_b), # nt 384 > cores
]

HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)


def timed(fn, reps=30, iters=8):
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
    return best / reps * 1e6


comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
for label, K, N, wdt in SHAPES:
    x = ttnn.from_torch(torch.randn(1, 1, 1, K) * 0.05, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.02, dtype=wdt,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    a64 = ttnn.to_torch(x, mesh_composer=comp)[:1].to(torch.float64)
    w64 = ttnn.to_torch(w, mesh_composer=comp)[:1].to(torch.float64)
    truth = a64.reshape(-1, K)[:1] @ w64.reshape(K, N)
    scale = truth.abs().max().item()
    kt, nt = K // 32, N // 32

    base = timed(lambda x=x, w=w: ttnn.linear(x, w, compute_kernel_config=HIFI4))
    print(f"RESULT {label}: default {base:7.2f}us  (kt {kt}, nt {nt}, {N_CORES} cores)",
          flush=True)

    per_core_min = (nt + N_CORES - 1) // N_CORES
    for per_core_N in (per_core_min, per_core_min * 2):
        if (nt + per_core_N - 1) // per_core_N > N_CORES:
            continue
        for in0_block_w in (2, 4, 5, 8, 10, 16, 20, 32):
            if kt % in0_block_w or in0_block_w > kt:
                continue
            try:
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=grid,
                    in0_block_w=in0_block_w,
                    out_subblock_h=1,
                    out_subblock_w=min(per_core_N, 4),
                    per_core_M=1,
                    per_core_N=per_core_N,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                got = ttnn.to_torch(
                    ttnn.linear(x, w, program_config=pc, compute_kernel_config=HIFI4),
                    mesh_composer=comp)[:1]
                us = timed(lambda x=x, w=w, pc=pc: ttnn.linear(
                    x, w, program_config=pc, compute_kernel_config=HIFI4))
            except Exception as exc:                                  # noqa: BLE001
                msg = (str(exc) or repr(exc)).splitlines()[0][:70]
                print(f"RESULT    N/core {per_core_N} blk {in0_block_w:3d}: {msg}",
                      flush=True)
                continue
            err = (got.reshape(-1, N)[:1].to(torch.float64) - truth).abs().max().item()
            print(f"RESULT    N/core {per_core_N} blk {in0_block_w:3d}: {us:7.2f}us "
                  f"{base/us:5.2f}x  err {err/max(scale,1e-30):.2e}", flush=True)
    ttnn.deallocate(x)
    ttnn.deallocate(w)

ttnn.close_mesh_device(mesh)
