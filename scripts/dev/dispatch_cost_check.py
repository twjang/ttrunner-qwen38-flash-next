"""What is one eager dispatch actually worth on the prefill path?

    uv run python scripts/dev/dispatch_cost_check.py [tokens]      (default 128)

5.7 asks for fewer device launches. Every estimate of what that would buy has
come from a *quoted* per-dispatch cost -- `op_count.py` prints "at ~0.30 ms an
eager dispatch", which for an 11681-call chunk predicts 3.50 s of dispatch in a
chunk that takes ~0.92 s wall clock. A cost model that predicts four times the
whole measurement is not a basis for deciding what to optimise, and no call site
in the by-caller table is above 4.9 % anyway, so the question worth answering is
not *which* op to cut but whether cutting any is worth the change.

So measure the slope instead of quoting a constant: wrap `gated_residual_mix`,
which runs a known number of times per chunk, so that each call issues `k` extra
tiny ops, and time the chunk at several `k`. The added ops do no useful work and
touch a 32x32 tile, so their device time is negligible and essentially all of
what they add is the dispatch itself. The gradient of wall clock against added
dispatches is the per-dispatch cost, measured on the path in question rather
than borrowed from another one.

The control that separates dispatch from compute is the *size* of the injected
op. The sweep runs twice, once on a 32x32 tile and once on a 128x2560 activation
-- 640 times the data. If the two slopes agree, what is being measured is the
launch and not the arithmetic, which is the claim "dispatch-bound" actually
makes; if the large one is steeper, some of the chunk is real compute and cutting
launches buys less than the slope suggests.

Warm-up then median, because a single cold sample has been wrong here twice
(`moe_chunk_sweep.py` and `bench_server.py` both had to be fixed for it).
"""
import statistics
import sys
import time

import ttnn

import twtest.tt.ops as ops
from _device_model import open_model

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 128
KS = (0, 2, 4, 8)
REPS = 5

mesh, cfg, m = open_model(max_seq_len=512)

_real = ops.gated_residual_mix
calls = 0
extra_k = 0
pad = None


def counting(*a, **kw):
    global calls
    calls += 1
    out = _real(*a, **kw)
    for _ in range(extra_k):
        inject()
    return out


ops.gated_residual_mix = counting
# model.py imported the name directly, so rebind it there too
import twtest.tt.model as model_mod
if getattr(model_mod, "gated_residual_mix", None) is _real:
    model_mod.gated_residual_mix = counting

import torch


def make_pad(rows, cols):
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, cols),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


PADS = {"32x32 tile": make_pad(32, 32), "128x2560 activation": make_pad(128, 2560)}
pad = PADS["32x32 tile"]

# The third arm asks a different question: `op_count.py` counts *python calls
# into ttnn*, which is not the same as device dispatches. `reshape` is 1094 of
# the 11681 it reports for a chunk and is frequently a host-side view. If
# injecting reshapes has no slope, then the counted total overstates the
# dispatches and the gap between the extrapolation and the measurement is
# explained rather than left hanging.
INJECT = {
    "32x32 tile": lambda: ttnn.add(pad, pad),
    "128x2560 activation": lambda: ttnn.add(pad, pad),
    "reshape (host-side view?)": lambda: ttnn.reshape(pad, tuple(pad.shape)),
}

prompt = [1000] * TOKENS


def once():
    st = m.new_state(batch=1)
    t0 = time.perf_counter()
    m.prefill(prompt, st)
    ttnn.synchronize_device(mesh)
    return 1000 * (time.perf_counter() - t0)


once()                       # warm: allocations and JIT out of the way
calls = 0
once()
per_chunk_calls = calls
print(f"RESULT gated_residual_mix runs {per_chunk_calls} times per {TOKENS}-token prefill",
      flush=True)

slopes = {}
inject = INJECT["32x32 tile"]
for label, fn in INJECT.items():
    pad = PADS.get(label, PADS["32x32 tile"])
    inject = fn
    rows = []
    print(f"RESULT --- injecting a {label} op ---", flush=True)
    for k in KS:
        extra_k = k
        ts = sorted(once() for _ in range(REPS))
        med = statistics.median(ts)
        added = k * per_chunk_calls
        rows.append((k, added, med))
        print(f"RESULT   k={k}  +{added:5d} dispatches   {med:8.1f} ms   "
              f"(min {ts[0]:.1f} max {ts[-1]:.1f})", flush=True)
    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    slopes[label] = (slope, rows[0][2])
    print(f"RESULT   per-dispatch: {1000 * slope:.1f} us", flush=True)

extra_k = 0
nop, _ = slopes["reshape (host-side view?)"]
print(f"RESULT --- an injected reshape costs {1000 * nop:.1f} us ---", flush=True)
small, base = slopes["32x32 tile"]
large, _ = slopes["128x2560 activation"]
print(f"RESULT --- 32x32 {1000 * small:.1f} us vs 128x2560 {1000 * large:.1f} us "
      f"({large / small:.2f}x for 640x the data) ---", flush=True)
print(f"RESULT a 128-token chunk issues 11681 calls; at the small-op slope that is "
      f"{11681 * small:.0f} ms against {base:.0f} ms measured "
      f"({100 * 11681 * small / base:.0f}%)", flush=True)
print(f"RESULT cutting the largest call site (576 calls, 4.9%) would save at most "
      f"{576 * small:.0f} ms ({100 * 576 * small / base:.1f}%)", flush=True)
ttnn.close_mesh_device(mesh)
