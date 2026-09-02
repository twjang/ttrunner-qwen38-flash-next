"""Where do a step's device ops go?

    uv run python scripts/dev/op_count.py [top]                  (default 25)

The single-user step is dispatch-bound -- 6355 ops at ~36 us traced -- so the
only thing that shortens it is issuing fewer. This wraps every callable in the
`ttnn` namespace with a counter and runs one eager step, which says which ops to
go after and, just as usefully, which are already negligible.

Counting, not profiling: an instrumented profile inflates totals ~40 % with
per-section syncs, and the question here is "how many", not "how long".
"""
import sys

import ttnn

from _device_model import open_model

TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 25
counts: dict[str, int] = {}


def wrap(ns, prefix=""):
    for name in dir(ns):
        if name.startswith("_"):
            continue
        try:
            fn = getattr(ns, name)
        except Exception:
            continue
        if not callable(fn) or isinstance(fn, type):
            continue
        key = f"{prefix}{name}"

        def made(fn=fn, key=key):
            def counted(*a, **kw):
                counts[key] = counts.get(key, 0) + 1
                return fn(*a, **kw)

            return counted

        try:
            setattr(ns, name, made())
        except Exception:
            pass


mesh, cfg, m = open_model(max_seq_len=512)
st = m.new_state(batch=1)
m.step([1000], st)          # warm: allocations and JIT out of the way
counts.clear()
wrap(ttnn)
wrap(ttnn.transformer, "transformer.")
wrap(ttnn.experimental, "experimental.")
m.step([1000], st)
total = sum(counts.values())
print(f"RESULT total device calls in one step: {total}", flush=True)
for name, n in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP]:
    print(f"RESULT {n:6d}  {100 * n / total:5.1f}%  {name}", flush=True)
print(f"RESULT per layer (48): {total / 48:.1f}", flush=True)
ttnn.close_mesh_device(mesh)
