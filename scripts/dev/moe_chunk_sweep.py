"""What should `moe_chunk` be? The tradeoff was reasoned about, not measured.

    uv run python scripts/dev/moe_chunk_sweep.py [chunk] [moe_chunk ...]
                                          (default 128, 8 16 32 64 128)

`TTModel.prefill` splits each chunk's MoE into `moe_chunk`-sized pieces because
the broadcast formulation computes |union of selected experts| x M rows, and the
union approaches all 512 as M grows -- M=128 wastes ~51x the FLOPs, M=16 about
8x. Smaller pieces waste less compute and issue more dispatches, and a
128-token chunk is dispatch-bound (19733 device calls, `op_count.py --prefill
128`), so the cheaper-looking end of that trade is not obviously the faster one.

Wall clock decides one half of it. The other half is that `moe_chunk` is *not*
a pure speed knob, however much the reasoning that it must be -- the split is
over rows, each row selects its own experts -- invites skipping the check: 8, 16
and 32 agree on every token while 64 and 128 degrade monotonically. So
`TTModel.prefill` caps it at `_MAX_MOE_CHUNK`, and this script reports rather
than raises above the cap, since re-deriving it is what the script is for.
Pair it with `device_quality.py --prefill 128 --moe-chunk N`.
"""
import sys
import time

import ttnn

from _device_model import open_model

args = [int(x) for x in sys.argv[1:]]
CHUNK = args[0] if args else 128
MOE = args[1:] or [8, 16, 32, 64, 128]

mesh, cfg, m = open_model(max_seq_len=512)
prompt = [1000] * CHUNK

best = None
for moe_chunk in MOE:
    if moe_chunk > CHUNK:
        continue
    try:
        m.prefill(prompt, m.new_state(batch=1), chunk=CHUNK, moe_chunk=moe_chunk)  # warm
    except ValueError as exc:
        print(f"RESULT moe_chunk {moe_chunk:4d}  refused: {exc}", flush=True)
        continue
    st = m.new_state(batch=1)
    t0 = time.perf_counter()
    m.prefill(prompt, st, chunk=CHUNK, moe_chunk=moe_chunk)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) * 1000
    print(f"RESULT moe_chunk {moe_chunk:4d}  {dt:8.1f} ms  "
          f"{CHUNK / (dt / 1000):7.1f} tok/s", flush=True)
    if best is None or dt < best[1]:
        best = (moe_chunk, dt)
print(f"RESULT best moe_chunk={best[0]} at {best[1]:.1f} ms for a {CHUNK}-token chunk", flush=True)
ttnn.close_mesh_device(mesh)
