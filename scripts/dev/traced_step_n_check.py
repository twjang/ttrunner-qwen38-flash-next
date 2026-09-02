"""Does a captured `step_n` reproduce the eager one, and what does it cost?

    uv run python scripts/dev/traced_step_n_check.py [k ...]      (default 2 4 8)

`step_n` only earns its place once it is traced: at k=4 it costs 864 ms eager
against 2071 ms for four eager steps, but four *traced* steps are 944 ms -- so an
untraced verifier is beaten by the thing it replaces. This measures the captured
one against both.

Correctness first, as always: the tokens a replay produces must be the tokens
the eager path produces from the same state.
"""
import os
import sys
import time

import ttnn

from _device_model import open_model, synthetic_prompt

from twtest.tt.traced import TracedDecoder, TracedStepN

KS = [int(x) for x in sys.argv[1:]] or [2, 4, 8]
PRE = 8
SEQ = int(os.environ.get("TWTEST_MAX_SEQ", "512"))
mesh, cfg, m = open_model(max_seq_len=SEQ)
print(f"RESULT max_seq_len {SEQ}", flush=True)
prompt = synthetic_prompt(PRE + max(KS) + 4)

# the baseline: a traced single-token step
st = m.new_state(batch=1)
dec = TracedDecoder(m, st)
dec.reset()
for t in prompt[:PRE]:
    dec.step([t])
ttnn.synchronize_device(mesh)
runs = []
for _ in range(10):
    t0 = time.perf_counter()
    dec.step([1000])
    ttnn.synchronize_device(mesh)
    runs.append(time.perf_counter() - t0)
runs.sort()
step_ms = 1000 * runs[len(runs) // 2]
print(f"RESULT traced step {step_ms:.1f} ms (only trace live)", flush=True)

# Does a second live capture slow the first one's replay? The engine keeps both
# -- a decoder for ordinary rounds and a step_n for verification -- and its
# ordinary rounds measured about twice the baseline even on a workload where the
# drafter almost never fires.
tr2 = TracedStepN(m, st, KS[0])
ttnn.synchronize_device(mesh)
runs = []
for _ in range(10):
    t0 = time.perf_counter()
    dec.step([1000])
    ttnn.synchronize_device(mesh)
    runs.append(time.perf_counter() - t0)
runs.sort()
both_ms = 1000 * runs[len(runs) // 2]
print(f"RESULT traced step {both_ms:.1f} ms (step_n capture also live)  "
      f"ratio {both_ms / step_ms:.2f}x", flush=True)
tr2.release()
dec.release()
del st, dec, tr2

for k in KS:
    draft = prompt[PRE : PRE + k]

    st = m.new_state(batch=1)
    for t in prompt[:PRE]:
        m.step([t], st)
    eager_tok = m.greedy_tokens(m.step_n(draft, st))[:k]
    del st

    st = m.new_state(batch=1)
    tr = TracedStepN(m, st, k)
    # rewind everything the two warmups and the capture consumed
    for layer in st.layers:
        for name in ("recurrent", "conv", "ple_conv", "keys", "values"):
            buf = getattr(layer, name)
            if buf is None:
                continue
            for entry in (buf if isinstance(buf, list) else [buf]):
                ttnn.copy(ttnn.zeros(list(entry.shape), dtype=entry.dtype,
                                     layout=entry.layout, device=mesh), entry)
        layer.conv_step = 0
        layer.ple_step = 0
    st.positions = [0]
    st.histories = [[]]
    for t in prompt[:PRE]:
        m.step([t], st)
    out = tr.step_n(draft)
    traced_tok = m.greedy_tokens(out)[:k]

    ttnn.synchronize_device(mesh)
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        tr.step_n(draft)
        ttnn.synchronize_device(mesh)
        runs.append(time.perf_counter() - t0)
    runs.sort()
    ms = 1000 * runs[len(runs) // 2]
    tr.release()
    del st, tr

    print(
        f"RESULT k={k:2d}  {'MATCH' if traced_tok == eager_tok else f'DIFFER {traced_tok} vs {eager_tok}'}"
        f"   traced step_n {ms:7.1f} ms   {k} traced steps {k * step_ms:7.1f} ms   "
        f"speedup {k * step_ms / ms:5.2f}x   per token {ms / k:6.1f} ms",
        flush=True,
    )
ttnn.close_mesh_device(mesh)
