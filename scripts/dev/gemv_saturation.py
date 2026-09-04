"""Do our decode matmuls saturate DRAM? Measured per shape, at M=1, 32 and 512.

    uv run python scripts/dev/gemv_saturation.py

The component ablations were the wrong frame. They account for 113.5 ms of a
146.2 ms step and every hunt for the rest came back small -- and then the host
split showed the host is 2.56 ms and the device replay is 141.70 ms. So the
answer is not a missing component, it is that the whole step runs slowly.

The arithmetic that reframes it: the model must read 2.92 GB per device per
token (dense 2.18 + top-10 experts 0.52 + head 0.22). At 141.7 ms that is
**20.6 GB/s effective**, against 273 GB/s measured on bfloat4_b and 388 GB/s
reached by a hand-written data-movement kernel. Five per cent of the machine.

A GEMV has arithmetic intensity ~1 op/byte, so it can only ever be memory bound;
if it is running at 5 % of bandwidth then it is neither compute nor memory bound
but bound by how the op is scheduled -- too few cores, short bursts, or a
blocking chosen for GEMM shapes.

This prices that directly, on the actual weight shapes of this model, inside a
trace so dispatch is not in the way. If M=1 lands near 20 GB/s and M=512 near
273, the problem is the row count and the fix is either a program config that
shards the weight across cores or our own kernel -- `ttnn.generic_op` is proven
(handoff 4g).
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                # noqa: E402

# (label, K, N) -- the shapes the decode path actually issues, per device.
# attn_qkv and ssm_out are column-sharded across 4 devices; hc_* are replicated.
SHAPES = [
    ("attn_qkv      ", 2560, 4608),
    ("attn_output   ", 6144, 2560),
    ("ssm_out       ", 4096, 2560),
    ("hc_down (rep) ", 10240, 640),
    ("hc_up   (rep) ", 640, 10240),
    ("router  (rep) ", 2560, 512),
    ("lm_head       ", 2560, 62080),
]
ROWS = [1, 32, 128, 512]
REPS = 20          # linears per trace, so one replay amortises the launch
ITERS = 10

mesh, cfg, m = open_model(max_seq_len=512)
HIFI4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4) \
    if hasattr(ttnn, "WormholeComputeKernelConfig") else None

print("RESULT shape           rows    ms/call   GB/s   %of 388", flush=True)

for label, K, N in SHAPES:
    w = ttnn.from_torch(
                (torch.randn(1, 1, K, N) * 0.02),
                dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    wbytes = K * N * 0.5625          # bfloat4_b: 16 nibbles + a shared exponent per 16
    for rows in ROWS:
        x = ttnn.from_torch(
                    (torch.randn(1, 1, rows, K) * 0.05),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        try:
            for _ in range(2):
                ttnn.linear(x, w)                   # compile
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            for _ in range(REPS):
                ttnn.linear(x, w)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            best = float("inf")
            for _ in range(ITERS):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                best = min(best, time.perf_counter() - t0)
            ttnn.release_trace(mesh, tid)
            per = best / REPS
            gbs = wbytes / per / 1e9
            print(f"RESULT {label} {rows:5d} {per * 1e3:9.4f} {gbs:7.1f} {100 * gbs / 388:7.1f}",
                  flush=True)
        except Exception as exc:                                    # noqa: BLE001
            print(f"RESULT {label} {rows:5d}   rejected: {type(exc).__name__}: "
                  f"{(str(exc) or repr(exc)).splitlines()[0][:110]}", flush=True)
        ttnn.deallocate(x)
    ttnn.deallocate(w)

print("RESULT ---", flush=True)
print("RESULT for scale: the whole step reads 2.92 GB/device in 141.7 ms = 20.6 GB/s",
      flush=True)
print("RESULT bandwidth measured elsewhere on this box: 273 GB/s bfloat4_b, "
      "388 GB/s by a custom DM kernel", flush=True)

ttnn.close_mesh_device(mesh)
