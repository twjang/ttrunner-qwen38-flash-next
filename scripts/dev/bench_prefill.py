"""Chunked-prefill wall clock at the default settings, with warmup and a median.

    uv run python scripts/dev/bench_prefill.py [tokens] [reps]   (default 128, 7)

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

mesh, cfg, m = open_model(max_seq_len=512)
prompt = [1000] * TOKENS


def once():
    st = m.new_state(batch=1)
    t0 = time.perf_counter()
    m.prefill(prompt, st)
    ttnn.synchronize_device(mesh)
    return 1000 * (time.perf_counter() - t0)


once()
ts = sorted(once() for _ in range(REPS))
med = statistics.median(ts)
print(f"RESULT prefill {TOKENS} tokens: median {med:7.1f} ms  min {ts[0]:7.1f}  "
      f"max {ts[-1]:7.1f}   {1000 * TOKENS / med:.1f} tok/s  "
      f"{med / TOKENS:.2f} ms/token", flush=True)
ttnn.close_mesh_device(mesh)
