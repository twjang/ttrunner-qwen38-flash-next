"""Where do a step's device ops go?

    uv run python scripts/dev/op_count.py [top] [--prefill N] [--by-caller]

The single-user step is dispatch-bound -- 6355 ops at ~36 us traced -- so the
only thing that shortens it is issuing fewer. This wraps every callable in the
`ttnn` namespace with a counter and runs one eager step, which says which ops to
go after and, just as usefully, which are already negligible.

Counting, not profiling: an instrumented profile inflates totals ~40 % with
per-section syncs, and the question here is "how many", not "how long".

`--by-caller` attributes each call to the `twtest` line that issued it, which is
what actually says where to cut -- "multiply, 3241" does not.

`--prefill N` counts one N-token chunked prefill instead. Multiply by ~57 us --
the measured value of a removed call on that path -- to price a reduction, and
treat the result as an upper bound, since some counted calls are host-side views
that never dispatch. `dispatch_cost_check.py` is where those numbers come from.
"""
import sys
import traceback

import ttnn

from _device_model import open_model

argv = sys.argv[1:]
BY_CALLER = "--by-caller" in argv
argv = [a for a in argv if a != "--by-caller"]
PREFILL = 0
if "--prefill" in argv:
    i = argv.index("--prefill")
    PREFILL = int(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
TOP = int(argv[0]) if argv else 25
counts: dict[str, int] = {}
by_caller: dict[str, int] = {}


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
                if BY_CALLER:
                    # the innermost twtest frame is the line that issued this op
                    for fr in reversed(traceback.extract_stack()[:-1]):
                        if "/twtest/" in fr.filename and "op_count" not in fr.filename:
                            site = f"{fr.filename.split('/twtest/')[-1]}:{fr.lineno} {fr.name}"
                            by_caller[site] = by_caller.get(site, 0) + 1
                            break
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
    # 57 us, not the 0.30 ms this used to print. That figure came from removing
    # 97 particular ops from an eager *decode* step and does not transfer: it
    # predicted 3.50 s of dispatch inside a chunk that measures 0.92 s. See
    # `dispatch_cost_check.py`, which measures the slope directly on this path,
    # and note the estimate below is still an overstatement because some of
    # these calls never dispatch -- an injected `reshape` costs 2.1 us.
    print(f"RESULT at ~57 us a removed call: {total * 0.057 / 1000:.2f} s, "
          f"an upper bound on what removing every one would buy", flush=True)
for name, n in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP]:
    print(f"RESULT {n:6d}  {100 * n / total:5.1f}%  {name}", flush=True)
print(f"RESULT per layer (48): {total / 48:.1f}", flush=True)
if BY_CALLER:
    print("RESULT --- by call site ---", flush=True)
    for site, n in sorted(by_caller.items(), key=lambda kv: -kv[1])[:TOP]:
        print(f"RESULT {n:6d}  {100 * n / total:5.1f}%  {site}", flush=True)
ttnn.close_mesh_device(mesh)
