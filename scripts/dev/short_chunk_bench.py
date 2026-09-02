"""Is consuming k tokens in one chunked pass cheaper than k decode steps?

    uv run python scripts/dev/short_chunk_bench.py [k ...]    (default 1 2 4 8 16 32)

This is the measurement that decides how speculation should verify. The handoff
proposes a `step_n` that unrolls the DeltaNet recurrence inside a batched step,
but the chunked path already consumes k tokens with the right causality and is
verified (87.5 % next-token top-1 against a same-positions control of 86.7 %),
so if a short chunk is cheap enough it is the same capability for far less code.

The DeltaNet op fixes its chunk at 128 and `prepare()` pads up to it, so a
2-token pass does most of a 128-token pass's work -- which is exactly what makes
this worth measuring rather than assuming.
"""
import sys
import time

import ttnn

from _device_model import open_model, synthetic_prompt

KS = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8, 16, 32]
mesh, cfg, m = open_model(max_seq_len=512)

# a decode step, for the baseline
st = m.new_state(batch=1)
for t in synthetic_prompt(4):
    m.step([t], st)
ttnn.synchronize_device(mesh)
samples = []
for _ in range(10):
    t0 = time.perf_counter()
    m.step([1000], st)
    ttnn.synchronize_device(mesh)
    samples.append(time.perf_counter() - t0)
samples.sort()
step_ms = 1000 * samples[len(samples) // 2]
print(f"RESULT one decode step {step_ms:.1f} ms", flush=True)
del st

for k in KS:
    ids = synthetic_prompt(k)
    # warm
    stw = m.new_state(batch=1)
    m.prefill(ids, stw, chunk=32)
    ttnn.synchronize_device(mesh)
    del stw
    runs = []
    for _ in range(5):
        stt = m.new_state(batch=1)
        t0 = time.perf_counter()
        m.prefill(ids, stt, chunk=32)
        ttnn.synchronize_device(mesh)
        runs.append(time.perf_counter() - t0)
        del stt
    runs.sort()
    chunk_ms = 1000 * runs[len(runs) // 2]
    print(
        f"RESULT k={k:3d}  chunked {chunk_ms:8.1f} ms   "
        f"{k} steps {k * step_ms:8.1f} ms   speedup {k * step_ms / chunk_ms:5.2f}x",
        flush=True,
    )
ttnn.close_mesh_device(mesh)
