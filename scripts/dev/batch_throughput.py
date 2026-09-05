"""Decode throughput against batch size -- tokens a second, not ms a step.

    uv run python scripts/dev/batch_throughput.py [max_batch]

The goal is throughput comparable to a 5090, which for this model is 32.6 ms a
token or 30.7 tokens a second. Single-stream decode is 51 ms, and this session
has established why that floor is where it is: 5764 ttnn ops at 5.8 us apiece is
33 ms before any arithmetic, and an op costs the same whatever its shape
(invariant 66).

That last part is the opening. If an op costs the same at one row as at
thirty-two, then a step serving B sequences costs barely more than a step
serving one, and throughput is B times better while each sequence's latency is
unchanged. `step_n_one.py` already shows the shape of it -- 8 rows in 187 ms --
but that is the speculative verifier, which unrolls the convolution and the
recurrence per token. Ordinary batched decode does not, so it should be cheaper.

Measured on the traced path, because that is what the engine runs.
"""
import sys
import time

import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

MAX_B = int(sys.argv[1]) if len(sys.argv) > 1 else 32
BATCHES = [b for b in (1, 2, 4, 8, 16, 32) if b <= MAX_B]

mesh, cfg, m = open_model(max_seq_len=2048)
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder        # noqa: E402

print(f"RESULT target: 32.6 ms a token, 30.7 tokens a second", flush=True)
print(f"RESULT {'batch':>5s} {'ms a step':>10s} {'ms a token':>11s} {'tok/s':>8s} "
      f"{'vs target':>10s}", flush=True)
for b in BATCHES:
    state = m.new_state(batch=b)
    try:
        dec = TracedDecoder(m, state)
        dec.reset()
        toks = [1000 + i for i in range(b)]
        for _ in range(3):
            dec.step(toks)
        ttnn.synchronize_device(mesh)
        best = float("inf")
        for _ in range(7):
            t0 = time.perf_counter()
            dec.step(toks)
            ttnn.synchronize_device(mesh)
            best = min(best, 1000 * (time.perf_counter() - t0))
        per = best / b
        print(f"RESULT {b:5d} {best:10.2f} {per:11.2f} {1000/per:8.1f} "
              f"{32.6/per:9.2f}x", flush=True)
        dec.release()
    except Exception as exc:                                          # noqa: BLE001
        print(f"RESULT {b:5d}: {type(exc).__name__}: "
              f"{(str(exc) or repr(exc)).splitlines()[0][:110]}", flush=True)

ttnn.close_mesh_device(mesh)
