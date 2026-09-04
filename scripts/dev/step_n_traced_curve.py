"""Does traced `step_n(k)` really cost `162 + 12.8k` ms? Measured, k = 1..8.

    uv run python scripts/dev/step_n_traced_curve.py

Every cost in this model is per call rather than per token: TILE_LAYOUT pads M=1
to 32 rows, and a fixed weight takes the same time at M=1 and M=32 (measured, in
`tiny_op_cost.py`). So verifying k drafted tokens in one step should cost barely
more than verifying one, and the per-token price should fall like 1/k.

The roofline audit fitted `t ~= 162 + 12.8k` to a traced curve and concluded k=8
would be **33.1 ms/token with no kernel work at all** -- more than the six
deployed optimisations put together, which is why this is worth checking before
building anything on it.

Two things make this cheap and safe to run. Only **one** trace is captured per
k, so the alternation defect in `ttnn_bug_report/` is never touched. And
`step_n` refuses to run with the QSA selection on (model.py raises
NotImplementedError), so the mesh is opened below `indexer_budget` where the
selection is off anyway -- which is also the regime the engine ships in.

A fit that holds says: capture `step_n(k)` once, draft with the n-gram table,
and the fixed 162 ms is amortised k ways. A fit that does not hold says go back
to fusing layers.
"""
import sys
import time

import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                # noqa: E402

KS = [1, 2, 4, 8]
ITERS = 7

mesh, cfg, m = open_model(max_seq_len=2048)      # below indexer_budget, so no selection
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder, TracedStepN  # noqa: E402

print(f"RESULT use_indexer={m.use_indexer} (step_n refuses if this is True)", flush=True)

rows = []
for k in KS:
    state = m.new_state(batch=1)
    try:
        if k == 1:
            dec = TracedDecoder(m, state)
            dec.reset()
            run = lambda: dec.step([1000])                          # noqa: E731
        else:
            dec = TracedStepN(m, state, k)
            run = lambda: dec.step_n([1000] * k)                    # noqa: E731
        for _ in range(3):
            run()
        ttnn.synchronize_device(mesh)
        best = float("inf")
        for _ in range(ITERS):
            t0 = time.perf_counter()
            run()
            ttnn.synchronize_device(mesh)
            best = min(best, 1000 * (time.perf_counter() - t0))
        per = best / k
        rows.append((k, best, per))
        print(f"RESULT k={k:2d}: {best:8.2f} ms a step, {per:7.2f} ms a token", flush=True)
        dec.release()
    except Exception as exc:                                        # noqa: BLE001
        print(f"RESULT k={k:2d}: rejected: {type(exc).__name__}: "
              f"{(str(exc) or repr(exc)).splitlines()[0][:160]}", flush=True)

if len(rows) >= 2:
    (k0, t0, _), (k1, t1, _) = rows[0], rows[-1]
    slope = (t1 - t0) / (k1 - k0)
    fixed = t0 - slope * k0
    print("RESULT ---", flush=True)
    print(f"RESULT fit: t ~= {fixed:.1f} + {slope:.2f}k ms  "
          f"(the audit predicted 162 + 12.8k)", flush=True)
    for k in (8, 16, 32):
        print(f"RESULT extrapolated k={k:2d}: {(fixed + slope * k) / k:6.2f} ms a token",
              flush=True)
    print(f"RESULT for scale: the shipping step is 96.21 ms a token, "
          f"the target is 32.6", flush=True)

ttnn.close_mesh_device(mesh)
