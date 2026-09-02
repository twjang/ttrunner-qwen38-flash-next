"""Where do a step's device ops go?

    uv run python scripts/dev/op_count.py [top] [--prefill N]    (default 25)

The single-user step is dispatch-bound -- 6355 ops at ~36 us traced -- so the
only thing that shortens it is issuing fewer. This wraps every callable in the
`ttnn` namespace with a counter and runs one eager step, which says which ops to
go after and, just as usefully, which are already negligible.

Counting, not profiling: an instrumented profile inflates totals ~40 % with
per-section syncs, and the question here is "how many", not "how long".

`--prefill N` counts one N-token chunked prefill instead. That is the number
that says whether tracing chunked prefill is worth the paged-cache refactor it
needs: multiply by the ~0.30 ms an eager dispatch costs and compare against the
measured wall clock.
"""
import sys

import ttnn

from _device_model import open_model

argv = sys.argv[1:]
PREFILL = 0
if "--prefill" in argv:
    i = argv.index("--prefill")
    PREFILL = int(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
TOP = int(argv[0]) if argv else 25
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
work = (lambda s: m.prefill([1000] * PREFILL, s)) if PREFILL else (lambda s: m.step([1000], s))
work(st)                    # warm: allocations and JIT out of the way
counts.clear()
wrap(ttnn)
wrap(ttnn.transformer, "transformer.")
wrap(ttnn.experimental, "experimental.")
work(m.new_state(batch=1) if PREFILL else st)
total = sum(counts.values())
what = f"one {PREFILL}-token chunked prefill" if PREFILL else "one step"
print(f"RESULT total device calls in {what}: {total}", flush=True)
if PREFILL:
    print(f"RESULT at ~0.30 ms an eager dispatch: {total * 0.30 / 1000:.2f} s of dispatch",
          flush=True)
for name, n in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP]:
    print(f"RESULT {n:6d}  {100 * n / total:5.1f}%  {name}", flush=True)
print(f"RESULT per layer (48): {total / 48:.1f}", flush=True)
ttnn.close_mesh_device(mesh)
