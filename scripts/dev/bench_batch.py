"""Aggregate decode throughput by batch size.

    uv run python scripts/dev/bench_batch.py [batch ...]     (default 1 8 32 64)

The README's "97.4 tok/s at batch 64" was measured before the correctness fixes
in `docs/iterations/013` and `014` -- on a model that was emitting the wrong
token -- so it needs re-taking on the model that works. It is also the first
exercise of the paged K/V cache above one sequence: `_kv_page_table` hands each
slot a contiguous run of blocks, and nothing until now has run that with
batch > 1.

5 warmup + 25 samples, per invariant 7. Aggregate tok/s is `batch / step_time`:
every slot advances one token per step.
"""
import os
import sys
import time

import ttnn

from _device_model import open_model, synthetic_prompt

BATCHES = [int(x) for x in sys.argv[1:]] or [1, 8, 32, 64]
SEQ = int(os.environ.get("TWTEST_MAX_SEQ", "512"))
WARM, ITERS = 5, 25

mesh, cfg, m = open_model(max_seq_len=SEQ)
print(f"RESULT max_seq_len {SEQ}  fused_experts {m.fuse_expert_gate_up}", flush=True)
print(f"RESULT {'batch':>6} {'ms/step':>10} {'tok/s':>10} {'vs batch 1':>11}", flush=True)

base = None
for batch in BATCHES:
    try:
        st = m.new_state(batch=batch)
        prompt = synthetic_prompt(8)
        for t in prompt:
            m.step([t] * batch, st)
        ttnn.synchronize_device(mesh)
        runs = []
        for i in range(WARM + ITERS):
            t0 = time.perf_counter()
            m.step([1000] * batch, st)
            ttnn.synchronize_device(mesh)
            if i >= WARM:
                runs.append(time.perf_counter() - t0)
        runs.sort()
        ms = 1000 * runs[len(runs) // 2]
        tps = batch / (ms / 1000)
        base = base or tps
        print(f"RESULT {batch:6d} {ms:10.1f} {tps:10.2f} {tps / base:10.1f}x", flush=True)
        del st
    except Exception as exc:
        print(f"RESULT {batch:6d}  FAILED {' '.join(str(exc).split())[:150]}", flush=True)
ttnn.close_mesh_device(mesh)
