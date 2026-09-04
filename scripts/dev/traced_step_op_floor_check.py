"""Is the *traced* decode step bound by a per-op floor?

    uv run python scripts/dev/traced_step_op_floor_check.py

`gemv_bandwidth_check.py` finds a fixed ~33 us per matmul that no amount of
bandwidth explains: a 0.37 MB weight and a 4.42 MB weight take 0.033 and 0.061 ms,
and only past ~8 MB does the kernel reach the 273-393 GB/s the hardware can
stream. Our projections are 1-4 MB per device -- 4-bit and sharded four ways --
so they sit in the floor-dominated regime.

Which raises an arithmetic coincidence worth checking: a decode step issues 6095
ops, and 6095 x 28 us is 171 ms against a measured traced step of 173.

Invariant 19 says the traced step is *not* dispatch-bound, on the evidence that
removing 97 ops moved it 236.1 -> 236.0 ms. If a per-op floor survives into the
trace, that evidence is wrong and op count is the lever for decode as well as
prefill. Settle it the way `dispatch_cost_check.py` settled the eager path:
inject a known number of ops, capture, and read the slope.
"""
import statistics
import time

import ttnn

import ttrunner_qwen38_flash_next.tt.ops as ops
import ttrunner_qwen38_flash_next.tt.model as model_mod
from _device_model import open_model

KS = (0, 2, 4, 8)
ITERS = 15

mesh, cfg, m = open_model(max_seq_len=4096)
_real = ops.gated_residual_mix
calls = 0
extra = 0
pad = None


def counting(*a, **kw):
    global calls
    calls += 1
    out = _real(*a, **kw)
    for _ in range(extra):
        ttnn.add(pad, pad)
    return out


ops.gated_residual_mix = counting
if getattr(model_mod, "gated_residual_mix", None) is _real:
    model_mod.gated_residual_mix = counting

import torch  # noqa: E402

pad = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16,
                      layout=ttnn.TILE_LAYOUT, device=mesh,
                      mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder  # noqa: E402

rows = []
for k in KS:
    extra = k
    state = m.new_state(batch=1)
    calls = 0
    dec = TracedDecoder(m, state)      # capture, with k extra ops per call site
    dec.reset()
    per_step = calls
    for _ in range(5):
        dec.step([1000])
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        dec.step([1000])
        ttnn.synchronize_device(mesh)
        ts.append(1000 * (time.perf_counter() - t0))
    med = statistics.median(ts)
    added = k * per_step
    rows.append((added, med))
    print(f"RESULT k={k}  +{added:4d} ops in the trace   {med:7.2f} ms", flush=True)
    dec.release()
    del dec, state

xs = [r[0] for r in rows]
ys = [r[1] for r in rows]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
print(f"RESULT --- per traced op: {1000 * slope:.1f} us ---", flush=True)
print(f"RESULT 6095 ops x that = {6095 * slope:.0f} ms against a {rows[0][1]:.0f} ms step",
      flush=True)
ttnn.close_mesh_device(mesh)
