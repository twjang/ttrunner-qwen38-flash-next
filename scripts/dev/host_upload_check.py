"""How many bytes cross host -> device per decode step?

    uv run python scripts/dev/host_upload_check.py

Only two tensors are `Residency.HOST`: the 28.8 GB n-gram table and the token
embedding, both gathers where a step touches a handful of rows. Everything else
is uploaded once at start-up and stays resident. That is the design; this is the
measurement, because "we only push a few rows" is exactly the kind of claim that
turns out to be wrong by a factor of a thousand.

Wraps every host->device entry point and counts bytes and wall clock across a
traced step.
"""
import time

import torch
import ttnn

from _device_model import open_model

mesh, cfg, m = open_model(max_seq_len=4096)
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder  # noqa: E402

stats = {"bytes": 0, "calls": 0, "seconds": 0.0}
_from_torch = ttnn.from_torch


def counting_from_torch(tensor, *a, **kw):
    t0 = time.perf_counter()
    out = _from_torch(tensor, *a, **kw)
    dt = time.perf_counter() - t0
    if kw.get("device") is not None or (len(a) and a[0] is not None):
        stats["calls"] += 1
        stats["bytes"] += tensor.numel() * tensor.element_size()
        stats["seconds"] += dt
    return out


state = m.new_state(batch=1)
dec = TracedDecoder(m, state)
dec.reset()
for _ in range(3):
    dec.step([1000])
ttnn.synchronize_device(mesh)

ttnn.from_torch = counting_from_torch
N = 10
t0 = time.perf_counter()
for _ in range(N):
    dec.step([1000])
ttnn.synchronize_device(mesh)
wall = (time.perf_counter() - t0) / N
ttnn.from_torch = _from_torch

per = {k: v / N for k, v in stats.items()}
print(f"RESULT step {1000 * wall:7.2f} ms", flush=True)
print(f"RESULT host->device per step: {per['calls']:.1f} calls, "
      f"{per['bytes'] / 1024:.1f} KB, {1000 * per['seconds']:.3f} ms "
      f"({100 * per['seconds'] / wall:.2f} % of the step)", flush=True)

# and what is actually resident
try:
    from ttrunner_qwen38_flash_next.tt.weights import TTWeights  # noqa: F401
    print(f"RESULT weight entries resident on device: {len(m.w.entries)}", flush=True)
except Exception:
    pass
ttnn.close_mesh_device(mesh)
