"""A/B two settings of a flag in **one process**, so drift cannot decide it.

    uv run python scripts/dev/ab_step.py ops _NO_SMALL_EW
    uv run python scripts/dev/ab_step.py ops _NO_FUSED_ROPE

Every A/B in this session has been two processes and a subtraction, and the rig
drifts about a millisecond between runs -- which is the size of most of the
things being measured. Three pairs of `decode_ablation_check.py` gave +0.64,
+0.23 and -0.01 for the same change.

`marginal_op_cost.py` showed the way out: it builds and releases a `TracedDecoder`
several times in one process and its numbers are clean to a few hundredths. So
this does the same, flipping a module attribute between captures.

Takes the minimum of nine timed steps, twice each way, alternating, and reports
both orders so a warm-up trend is visible rather than hidden.
"""
import importlib
import sys
import time

import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

MODULE = sys.argv[1] if len(sys.argv) > 1 else "ops"
FLAG = sys.argv[2] if len(sys.argv) > 2 else "_NO_SMALL_EW"

mesh, cfg, m = open_model(max_seq_len=4096)
m.selection_active = False
mod = importlib.import_module(f"ttrunner_qwen38_flash_next.tt.{MODULE}")
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder       # noqa: E402

if not hasattr(mod, FLAG):
    raise SystemExit(f"{MODULE} has no attribute {FLAG}")
original = getattr(mod, FLAG)


def measure(off: bool) -> float:
    setattr(mod, FLAG, off)
    state = m.new_state(batch=1)
    dec = TracedDecoder(m, state)
    dec.reset()
    for _ in range(3):
        dec.step([1000])
    ttnn.synchronize_device(mesh)
    best = float("inf")
    for _ in range(9):
        t0 = time.perf_counter()
        dec.step([1000])
        ttnn.synchronize_device(mesh)
        best = min(best, 1000 * (time.perf_counter() - t0))
    dec.release()
    return best


ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

print(f"RESULT flipping {MODULE}.{FLAG} (its default is {original})", flush=True)
# The first capture in a process is reliably the slowest -- caches, first-touch
# allocation, the program cache filling -- and whichever side runs first wears
# that cost. One discarded measurement makes the two sides comparable; without
# it a change was credited or blamed for up to 1.8 ms of warm-up.
measure(False)
on, off = [], []
for i in range(ROUNDS):
    a = measure(False)      # flag False = the feature is ON
    b = measure(True)       # flag True  = the feature is OFF
    on.append(a)
    off.append(b)
    print(f"RESULT   round {i + 1}: on {a:7.2f} ms   off {b:7.2f} ms   "
          f"{b - a:+.2f}", flush=True)
setattr(mod, FLAG, original)

mo, mf = min(on), min(off)
print(f"RESULT best: on {mo:7.2f} ms, off {mf:7.2f} ms -> "
      f"{mf - mo:+.2f} ms for the feature", flush=True)
ttnn.close_mesh_device(mesh)
