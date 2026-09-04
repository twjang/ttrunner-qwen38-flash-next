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
        b = sum(nbytes(x) for x in a if hasattr(x, "shape"))
        for o in (out if isinstance(out, (list, tuple)) else [out]):
            if hasattr(o, "shape"):
                b += nbytes(o)
        moved[key] += b
        return out

    setattr(mod, name, wrapped)


mesh, cfg, m = open_model(max_seq_len=4096)

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

print("RESULT ---", flush=True)
print(f"RESULT bytes alone imply {total_bytes / 388e9 * 1e3:.1f} ms at 388 GB/s; "
      f"the step measures ~104 ms", flush=True)
print(f"RESULT at the 1.4 us traced dispatch floor, {total_calls} calls is "
      f"{total_calls * 1.4 / 1e3:.1f} ms", flush=True)

ttnn.close_mesh_device(mesh)
