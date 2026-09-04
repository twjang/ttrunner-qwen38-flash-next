"""If 32 rows cost what 1 row costs, what does a traced `step_n` cost against k?

    uv run python scripts/dev/traced_step_n_scaling_check.py [k ...]

`gemv_padding_cost_check.py` shows the matmuls take the same time for M=1 as for
M=32 -- decode uses a thirty-second of the row capacity, which is why prefill is
~23x faster per token. That capacity cannot be reclaimed by removing the padding
(at M=1 the op is fetching weights at 73.6 GB/s, nowhere near the arithmetic
limit); it can only be reclaimed by *filling* the rows.

`step_n` fills them: it advances k tokens in one pass. So the question is whether
a traced `step_n` at k=32 costs what a traced step at k=1 costs. If it does, the
ceiling for speculative decoding here is 32 tokens for the price of one.

One capture live at a time -- captured, measured, released, next -- because
*alternating* two traces hangs this build, not capturing several in sequence
(`trace_capture_cost_check.py`).
"""
import sys
import time

import ttnn

from _device_model import open_model, synthetic_prompt

KS = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8, 16, 32]
PRE = 8
mesh, cfg, m = open_model(max_seq_len=1024)
from ttrunner_qwen38_flash_next.tt.traced import TracedStepN  # noqa: E402

prompt = synthetic_prompt(PRE + max(KS) + 8)
base = None
print("RESULT    k   traced step_n   per token   vs k=1", flush=True)
for k in KS:
    state = m.new_state(batch=1)
    for t in prompt[:PRE]:
        m.step([t], state)
    runner = TracedStepN(m, state, k)
    feed = prompt[PRE : PRE + k]
    for _ in range(3):
        runner.step_n(feed)
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(9):
        t0 = time.perf_counter()
        runner.step_n(feed)
        ttnn.synchronize_device(mesh)
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    ms = ts[len(ts) // 2]
    base = base or ms
    print(f"RESULT {k:5d}   {ms:9.1f} ms   {ms / k:8.1f}   {ms / base:5.2f}x", flush=True)
    runner.release()
    del runner, state

ttnn.close_mesh_device(mesh)
