"""Can the decode trace be re-captured *after* a chunked prefill?

    uv run python scripts/dev/recapture_after_prefill_check.py [prompt] [gen]

Handoff 5.2: chunked prefill and a live decode trace corrupt each other, so the
engine turns the trace off whenever chunked prefill is on -- ingestion at
7.2 ms/token, but every generated token at 588 ms instead of 177. For an agent
workload, where a long prompt is followed by a few hundred tokens, that is the
wrong half to keep.

The corruption is about a prefill allocating while a trace is *live*. So do not
have one live: release the trace, prefill, capture again. Measured costs make
that plausible -- release 15 ms, capture 2.65 s, and a 200-token generation saves
200 x 411 ms = 82 s, so it pays from about seven tokens on.

The catch is that capturing consumes two warm-up tokens and so dirties the state
the prefill just built, and `TracedDecoder.reset()` fixes that by zeroing
everything -- which would throw the prompt away. `snapshot`/`restore` (5.4 ms,
already used by speculation) is the right primitive instead.

    release -> prefill -> snapshot -> capture -> restore -> decode

This checks the tokens that come out are the ones a plain eager prefill-and-step
produces. Anything less than identical means the scheme is unsound.
"""
import sys
import time

import ttnn

from _device_model import open_model

PROMPT = int(sys.argv[1]) if len(sys.argv) > 1 else 128
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 8

mesh, cfg, m = open_model(max_seq_len=4096, trace_region_bytes=128 << 20)
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder  # noqa: E402

prompt = [(1000 + 7 * i) % 30000 for i in range(PROMPT)]


def greedy_eager():
    st = m.new_state(batch=1)
    h = m.prefill(prompt, st)
    out = []
    for _ in range(GEN):
        t = int(m.logits(h)[0].float().argmax())
        out.append(t)
        h = m.step([t], st)
    return out


def greedy_recapture():
    st = m.new_state(batch=1)
    # the engine's startup capture, against a fresh state
    dec = TracedDecoder(m, st)
    dec.reset()
    # ---- a request arrives ----
    t0 = time.perf_counter()
    dec.release()
    t_rel = time.perf_counter() - t0

    t0 = time.perf_counter()
    h = m.prefill(prompt, st)
    ttnn.synchronize_device(mesh)
    t_pre = time.perf_counter() - t0

    t0 = time.perf_counter()
    snap = m.snapshot(st)
    t_snap = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = TracedDecoder(m, st)          # capture; consumes two warm-up tokens
    ttnn.synchronize_device(mesh)
    t_cap = time.perf_counter() - t0

    t0 = time.perf_counter()
    m.restore(st, snap)                 # put the prompt's state back
    t_res = time.perf_counter() - t0

    out = []
    t0 = time.perf_counter()
    for _ in range(GEN):
        t = int(m.logits(h)[0].float().argmax())
        out.append(t)
        h = dec.step([t])
    ttnn.synchronize_device(mesh)
    t_gen = (time.perf_counter() - t0) / GEN
    print(f"RESULT release {1000 * t_rel:6.1f} ms   prefill {1000 * t_pre:7.1f} ms   "
          f"snapshot {1000 * t_snap:6.1f} ms   capture {1000 * t_cap:7.1f} ms   "
          f"restore {1000 * t_res:6.1f} ms   traced gen {1000 * t_gen:6.1f} ms/token",
          flush=True)
    return out


ref = greedy_eager()
got = greedy_recapture()
print(f"RESULT eager      {ref}", flush=True)
print(f"RESULT recaptured {got}", flush=True)
print(f"RESULT identical: {'YES' if ref == got else 'NO'}", flush=True)
ttnn.close_mesh_device(mesh)
