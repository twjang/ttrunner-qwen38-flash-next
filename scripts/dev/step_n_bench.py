"""What does `step_n` cost, warmed, against k sequential steps?

    uv run python scripts/dev/step_n_bench.py [k ...]          (default 1 2 4 8 16)

Each k is a new set of kernel shapes, so a first call carries its JIT: the naive
measurement read 3103 ms for k=4 where the warmed figure is far lower. 3 warmup
calls per k, then the median of 5 -- the hygiene in `tt/bench.py`, for the same
reason it exists there.
"""
import sys
import time

import ttnn

from _device_model import open_model, synthetic_prompt

KS = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8, 16]
PRE = 8
mesh, cfg, m = open_model(max_seq_len=1024)
prompt = synthetic_prompt(PRE + max(KS) + 8)

st = m.new_state(batch=1)
for t in prompt[:PRE]:
    m.step([t], st)
for _ in range(5):
    m.step([1000], st)
ttnn.synchronize_device(mesh)
runs = []
for _ in range(10):
    t0 = time.perf_counter()
    m.step([1000], st)
    ttnn.synchronize_device(mesh)
    runs.append(time.perf_counter() - t0)
runs.sort()
step_ms = 1000 * runs[len(runs) // 2]
del st
print(f"RESULT one eager step {step_ms:.1f} ms", flush=True)

for k in KS:
    draft = prompt[PRE : PRE + k]

    def once():
        s = m.new_state(batch=1)
        for t in prompt[:PRE]:
            m.step([t], s)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        m.step_n(draft, s)
        ttnn.synchronize_device(mesh)
        dt = time.perf_counter() - t0
        del s
        return dt

    for _ in range(3):
        once()
    runs = sorted(once() for _ in range(5))
    ms = 1000 * runs[len(runs) // 2]
    print(
        f"RESULT k={k:3d}  step_n {ms:8.1f} ms   {k} steps {k * step_ms:8.1f} ms   "
        f"speedup {k * step_ms / ms:5.2f}x   per token {ms / k:7.1f} ms",
        flush=True,
    )
ttnn.close_mesh_device(mesh)
