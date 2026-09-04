"""Traced and eager single-user step time, with the usual hygiene.

    uv run python scripts/dev/bench_step.py [iters] [max_seq_len]
                                                     (default 25, 262144)

5 warmup + `iters` samples, per `tt/bench.py`. Run after any change to the hot
path: a cold single run once misreported a 1.3x win as 0.94x.
"""
import sys

import ttnn

from _device_model import open_model

from ttrunner_qwen38_flash_next.tt.bench import benchmark_step

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 262144
mesh, cfg, m = open_model(max_seq_len=SEQ)
print(f'RESULT max_seq_len {SEQ}', flush=True)

state = m.new_state(batch=1)
eager = benchmark_step(m, state, 1000, iters=ITERS)
print(f"RESULT eager  {eager}", flush=True)
del state

from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder  # noqa: E402

state = m.new_state(batch=1)
dec = TracedDecoder(m, state)
dec.reset()
import time  # noqa: E402

for _ in range(5):
    dec.step([1000])
ttnn.synchronize_device(mesh)
samples = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    dec.step([1000])
    ttnn.synchronize_device(mesh)
    samples.append(time.perf_counter() - t0)
samples.sort()
print(f"RESULT traced median {1000 * samples[len(samples) // 2]:.1f} ms  "
      f"min {1000 * samples[0]:.1f} ms", flush=True)
ttnn.close_mesh_device(mesh)
