"""One `step_n(k)`, timed, in its own process.

    uv run python scripts/dev/step_n_one.py <k> [--moe-stub]

`step_n_traced_curve.py` walks k inside a single process, and the bound input
buffers are keyed by name rather than by width -- so the [2] buffer built for
k=2 is handed a [4] tensor at k=4 and the run dies on a shape assert. That is a
harness artefact, not a model limit, and it hid the shape of the curve.

`--moe-stub` replaces `moe_block` with an identity, which says how much of the
k>1 cost is the MoE. It matters because the wide gather path is exact at one row
only (handoff 15.3), so `step_n` falls back to `sparse_matmul` -- and since the
experts moved to the intermediate axis that op's [1, E, M, K] zero-fill is four
times what it was.
"""
import sys
import time

import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 1
STUB = "--moe-stub" in sys.argv

mesh, cfg, m = open_model(max_seq_len=2048)
if STUB:
    import ttrunner_qwen38_flash_next.tt.moe as moe
    moe.moe_block = lambda mixed, *a, **kw: mixed

from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder, TracedStepN  # noqa: E402

state = m.new_state(batch=1)
if K == 1:
    dec = TracedDecoder(m, state)
    dec.reset()
    run = lambda: dec.step([1000])                                    # noqa: E731
else:
    dec = TracedStepN(m, state, K)
    run = lambda: dec.step_n([1000] * K)                              # noqa: E731

for _ in range(3):
    run()
ttnn.synchronize_device(mesh)
best = float("inf")
for _ in range(7):
    t0 = time.perf_counter()
    run()
    ttnn.synchronize_device(mesh)
    best = min(best, 1000 * (time.perf_counter() - t0))
print(f"RESULT k={K:2d}{' moe-stub' if STUB else '        '}: {best:8.2f} ms a step, "
      f"{best / K:7.2f} ms a token", flush=True)
dec.release()
ttnn.close_mesh_device(mesh)
