"""Chunked-prefill wall clock at the default settings, with warmup and a median.

    uv run python scripts/dev/bench_prefill.py [tokens] [reps] [chunk...]
                                                 (default 128, 7, 128)

Extra arguments sweep `prefill(chunk=)`. Prefill is dispatch-bound (invariant
21) and many of its ops are per-layer-per-chunk rather than per-row, so a wider
chunk should amortise them -- which is a speed question first and an accuracy
question only if it turns out to be worth something.

A single cold draw has misreported this path twice (5.8's first table, and
`bench_server.py`), so: one warm-up prefill to get allocations and JIT out of the
way, then a median.
"""
import statistics
import sys
import time

import ttnn

from _device_model import open_model

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 128
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
CHUNKS = [int(a) for a in sys.argv[3:]] or [128]

mesh, cfg, m = open_model(max_seq_len=512)
prompt = [1000] * TOKENS


def once(chunk):
    st = m.new_state(batch=1)
    t0 = time.perf_counter()
    m.prefill(prompt, st, chunk=chunk)
    ttnn.synchronize_device(mesh)
    return 1000 * (time.perf_counter() - t0)


for chunk in CHUNKS:
    once(chunk)
    ts = sorted(once(chunk) for _ in range(REPS))
    med = statistics.median(ts)
    print(f"RESULT prefill {TOKENS} tokens, chunk={chunk:4d}: median {med:7.1f} ms  "
          f"min {ts[0]:7.1f}  max {ts[-1]:7.1f}   {1000 * TOKENS / med:.1f} tok/s  "
          f"{med / TOKENS:.2f} ms/token", flush=True)
ttnn.close_mesh_device(mesh)
