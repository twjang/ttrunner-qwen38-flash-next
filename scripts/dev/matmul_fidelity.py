"""What does MathFidelity cost, and what does it buy, on the model's own shapes?

    uv run python scripts/dev/matmul_fidelity.py

`matmul_dtype_width.py` found that at M=1 these matmuls do not care about their
weight's dtype: [2560, 3200] is 61.4 us at bfloat4_b, bfloat8_b *and* bfloat16.
So they are not bandwidth-bound, and the cost tracks the length of the K loop --
roughly 0.5-0.8 us a k-tile across every shape in the census.

Which points at the FPU. Every matmul in this model runs at **HiFi4**, the
four-pass mode, and the operands are a bfloat16 activation against a bfloat8_b or
bfloat4_b weight -- 7 and 3 mantissa bits. HiFi4 exists for full bfloat16 x
bfloat16; on these operands the extra passes may be buying nothing at all.

The linears are 27 ms of a 59 ms step, so this is the largest single question
left in the model. Time *and* error against float64, because a faster wrong
answer is not a result (invariant 57).
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

# (label, K, N, weight dtype) -- the shapes that hold the linears' time.
SHAPES = [
    ("MoE gate|up   [2560, 3200]", 2560, 3200, ttnn.bfloat4_b),
    ("MoE down      [1600, 2560]", 1600, 2560, ttnn.bfloat8_b),
    ("shexp gate|up [2560, 1280]", 2560, 1280, ttnn.bfloat8_b),
    ("attn_gate     [2560, 1536]", 2560, 1536, ttnn.bfloat8_b),
    ("attn_output   [6144, 2560]", 6144, 2560, ttnn.bfloat8_b),
    ("hc_up          [320, 10240]", 320, 10240, ttnn.bfloat8_b),
]

FID = [
    ("HiFi4", ttnn.MathFidelity.HiFi4),
    ("HiFi3", ttnn.MathFidelity.HiFi3),
    ("HiFi2", ttnn.MathFidelity.HiFi2),
    ("LoFi", ttnn.MathFidelity.LoFi),
]


def cfg_for(f):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=f, math_approx_mode=False,
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
    # the truth, on the *quantised* operands the device actually holds
    a64 = ttnn.to_torch(x, mesh_composer=comp)[:1].to(torch.float64)
    w64 = ttnn.to_torch(w, mesh_composer=comp)[:1].to(torch.float64)
    truth = a64.reshape(-1, K)[:1] @ w64.reshape(K, N)
    scale = truth.abs().max().item()

    parts = []
    base = None
    for name, f in FID:
        c = cfg_for(f)
        try:
            got = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=c),
                                mesh_composer=comp)[:1]
            us = timed(lambda x=x, w=w, c=c: ttnn.linear(x, w, compute_kernel_config=c))
        except Exception as exc:                                      # noqa: BLE001
            parts.append(f"{name} {type(exc).__name__}")
            continue
        err = (got.reshape(-1, N)[:1].to(torch.float64) - truth).abs().max().item()
        rel = err / max(scale, 1e-30)
        base = base if base is not None else us
        parts.append(f"{name} {us:7.2f}us {base/us:4.2f}x err {rel:.2e}")
    print(f"RESULT {label}", flush=True)
    for p in parts:
        print(f"RESULT    {p}", flush=True)
    ttnn.deallocate(x)
    ttnn.deallocate(w)

ttnn.close_mesh_device(mesh)
