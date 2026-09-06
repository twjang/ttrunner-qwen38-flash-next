"""Every ttnn op one decode step issues, counted and sized by name.

    uv run python scripts/dev/step_op_census.py

The weight arithmetic says the step should not cost what it costs. Per device per
token the model reads ~9.5 GB... no: ~9.5 *ms* worth of dense weight and ~1.3 ms
of top-10 expert weight, about **11 ms of weight reading** against a 104 ms
step. So 93 ms is not weights.

Which leaves op count and intermediates, and the way to find out which is to
count. This wraps every ttnn callable the decode path touches, runs one eager
`model.step`, and reports calls and moved bytes by op name -- so "fuse this" can
be aimed rather than guessed.

Bytes are input + output for elementwise and data movement, and for matmuls the
weight plus the output, which is what those actually stream. It is an estimate
from shapes, not a profiler reading; the point is the ranking.
"""
import collections
import sys

import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                # noqa: E402

BYTES = {
    ttnn.bfloat4_b: 0.5625, ttnn.bfloat8_b: 1.0625,
    ttnn.bfloat16: 2.0, ttnn.float32: 4.0, ttnn.uint16: 2.0,
    ttnn.uint32: 4.0, ttnn.int32: 4.0,
}


def nbytes(t):
    try:
        n = 1
        for d in t.shape:
            n *= int(d)
        return n * BYTES.get(t.dtype, 2.0)
    except Exception:                                               # noqa: BLE001
        return 0.0


calls = collections.Counter()
moved = collections.Counter()
# Bucketed per *call* by the largest operand in tiles, which is what decides how
# many cores an op can use. Per call and not per op name: `ttnn.multiply` is
# issued on one tile and on three hundred and twenty, and the average of those
# is not a thing that exists.
BUCKETS = [(1, "1", 1), (4, "2-4", 4), (16, "5-16", 16), (32, "17-32", 32),
           (64, "33-64", 64), (10 ** 9, "65+", 110)]
per_call = collections.Counter()
per_call_name = collections.Counter()
# Where the small calls come from, so the work can be aimed at source lines
# rather than at op names. Only for calls whose largest operand is small, since
# those are the ones a right-sized launch would help.
sites = collections.Counter()

# Every ttnn name the decode path uses, by inspection of model.py / moe.py /
# linear_attn.py / ops.py. Wrapping the module attribute catches every call site
# because they all resolve `ttnn.<name>` at call time.
NAMES = [
    "linear", "matmul", "sparse_matmul", "add", "multiply", "subtract", "silu",
    "sigmoid", "relu", "exp", "reciprocal", "rsqrt", "sqrt", "tanh", "gelu",
    "reshape", "permute", "transpose", "slice", "concat", "repeat", "typecast",
    "to_memory_config", "to_layout", "copy", "clone", "zeros_like", "topk",
    "scatter", "le", "sum", "mean", "max", "argmax", "all_reduce",
    "mesh_partition", "all_gather", "embedding", "rms_norm", "layer_norm",
    "softmax", "pad", "tilize", "untilize", "split", "chunk",
]


def wrap(mod, name):
    fn = getattr(mod, name, None)
    if fn is None or not callable(fn):
        return
    key = f"{'ttnn' if mod is ttnn else mod.__name__.split('.')[-1]}.{name}"

    def wrapped(*a, **kw):
        out = fn(*a, **kw)
        calls[key] += 1
        big = 0
        for x in a:
            sh = getattr(x, "shape", None)
            if sh is None or len(sh) < 2:
                continue
            n = 1
            for d in list(sh)[:-2]:
                n *= d
            n *= max(1, (sh[-2] + 31) // 32) * max(1, (sh[-1] + 31) // 32)
            big = max(big, n)
        for _hi, _name, _c in BUCKETS:
            if big <= _hi:
                per_call[_name] += 1
                per_call_name[(key, _name)] += 1
                if True:
                    import traceback as _tb
                    for fr in reversed(_tb.extract_stack()[:-1]):
                        if "/tt/" in fr.filename and "census" not in fr.filename:
                            sites[(f"{fr.filename.split('/tt/')[-1]}:{fr.lineno}",
                                   key, big)] += 1
                            break
                break
        b = sum(nbytes(x) for x in a if hasattr(x, "shape"))
        for o in (out if isinstance(out, (list, tuple)) else [out]):
            if hasattr(o, "shape"):
                b += nbytes(o)
        moved[key] += b
        return out

    setattr(mod, name, wrapped)


mesh, cfg, m = open_model(max_seq_len=4096)
# `TracedDecoder` sets this before it captures, and the traced step is what the
# 32 ms is measured on -- but `TTModel.trace_safe_rings` defaults to False, so an
# eager census takes the host-rotated ring path and never reaches
# `fused_conv_step` (model.py:778) or the other two branches at 1907/2405. Left
# at the default this script counts a code path that does not ship.
# TTRUNNER_TRACE_RINGS=0 restores the eager path for comparison.
import os as _os
m.trace_safe_rings = _os.environ.get("TTRUNNER_TRACE_RINGS", "1") == "1"
print(f"RESULT trace_safe_rings {m.trace_safe_rings}", flush=True)

for n in NAMES:
    wrap(ttnn, n)
for sub in ("transformer", "experimental"):
    s = getattr(ttnn, sub, None)
    if s is not None:
        for n in ("scaled_dot_product_attention_decode",
                  "paged_scaled_dot_product_attention_decode",
                  "paged_update_cache", "update_cache"):
            wrap(s, n)

state = m.new_state(batch=1)
m.step([1000], state)          # warm: lazy weight loads and first-call paths
calls.clear()
moved.clear()
per_call.clear()
per_call_name.clear()
sites.clear()
m.step([1000], state)
ttnn.synchronize_device(mesh)

total_calls = sum(calls.values())
total_bytes = sum(moved.values())
print(f"RESULT one step: {total_calls} ttnn calls, "
      f"{total_bytes / 1e9:.2f} GB moved (per device, from shapes)", flush=True)
print(f"RESULT {'op':52s} {'calls':>7s} {'GB':>8s} {'ms@388':>8s} {'us/call':>8s}",
      flush=True)
for key, n in calls.most_common(26):
    gb = moved[key] / 1e9
    ms = gb * 1e9 / 388e9 * 1e3
    print(f"RESULT {key:52s} {n:7d} {gb:8.3f} {ms:8.2f} {ms * 1e3 / n:8.1f}", flush=True)

# --- how many of them are small enough to want fewer cores ------------------
#
# `dispatch_floor.py` prices a launch by its core count: a generic_op touching
# one page is 2.06 us on one core, 3.15 on thirty-two and 5.81 on a hundred and
# ten. `ttnn.multiply` is 5.78 whatever the tensor -- one tile or three hundred
# and twenty -- so ttnn takes the whole grid regardless, and an op on a handful
# of tiles pays a full-grid launch for nothing.
print("RESULT the widest calls (65+ tiles), where only fusion helps:", flush=True)
wide = [((site, key, t), n) for (site, key, t), n in sites.items() if t > 64]
wide.sort(key=lambda kv: -kv[1] * kv[0][2] ** 0)
for (site, key, t), n in sorted(wide, key=lambda kv: -kv[1])[:200]:
    print(f"RESULT   {site:34s} {key:20s} {t:5d}t {n:4d}x "
          f"{n * 5.33 / 1000:5.2f}ms", flush=True)

import json as _json
with open("/tmp/site_dump.json", "w") as _f:
    _json.dump([[site, key, t, n] for (site, key, t), n in sites.items()], _f)
print("RESULT wrote /tmp/site_dump.json", flush=True)
print("RESULT ---", flush=True)
print(f"RESULT {'largest operand, tiles':26s} {'calls':>7s} {'now':>9s} "
      f"{'own cores':>10s} {'saving':>8s}", flush=True)
by_bucket = per_call
total = 0.0
for hi, name, cores in BUCKETS:
    n = by_bucket.get(name, 0)
    if not n:
        continue
    cheap = 1.8 + 0.036 * cores
    save = max(0.0, n * (5.8 - cheap) / 1000)
    total += save
    print(f"RESULT {name:26s} {n:7d} {n * 5.8 / 1000:8.2f}ms "
          f"{n * cheap / 1000:9.2f}ms {save:7.2f}ms", flush=True)
print(f"RESULT {'-> every op on its own cores':26s} {'':7s} {'':9s} {'':10s} "
      f"{total:7.2f}ms", flush=True)
for bucket in ("1", "2-4", "5-16"):
    by = collections.Counter()
    for (key, name), n in per_call_name.items():
        if name == bucket:
            by[key] += n
    if not by:
        continue
    print(f"RESULT calls whose largest operand is {bucket} tile(s):", flush=True)
    for key, n in by.most_common(10):
        print(f"RESULT   {key:40s} {n:5d}  ({n * 3.7 / 1000:.2f} ms if right-sized)",
              flush=True)

print("RESULT ---", flush=True)
print(f"RESULT bytes alone imply {total_bytes / 388e9 * 1e3:.1f} ms at 388 GB/s; "
      f"the step measures ~104 ms", flush=True)
print(f"RESULT at the 1.4 us traced dispatch floor, {total_calls} calls is "
      f"{total_calls * 1.4 / 1e3:.1f} ms", flush=True)

ttnn.close_mesh_device(mesh)
