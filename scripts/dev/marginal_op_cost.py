"""What does one more op cost *in the model's own trace*?

    uv run python scripts/dev/marginal_op_cost.py

Every plan in this document has been priced off "an op is 5.8 us", measured by
timing sixty-four copies of one op in a trace of its own. Three changes built on
that have now under-delivered by roughly the same factor: the fused reinject
(0.4 ms against 1.6 predicted), the raw gate stream (0.2 against 1.85), and 469
right-sized elementwise calls (0.0-0.3 against 1.7).

So measure the thing directly: insert K extra one-tile `ttnn.multiply` calls into
each decoder layer and see what the step costs. The slope is the marginal price
of an op where it actually lives -- next to 5764 others, in a 48-layer trace --
which is the number every one of those plans needed and none of them had.
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

mesh, cfg, m = open_model(max_seq_len=4096)
m.selection_active = False
import ttrunner_qwen38_flash_next.tt.model as model_mod              # noqa: E402
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder       # noqa: E402

rep = ttnn.ReplicateTensorToMesh(mesh)
pad = ttnn.from_torch(torch.randn(1, 1, 32, 32), dtype=ttnn.bfloat16,
                      layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
WIDE = ttnn.from_torch(torch.randn(1, 1, 32, 10240), dtype=ttnn.bfloat16,
                       layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

real_layer = model_mod.TTModel._layer
EXTRA = {"n": 0, "t": pad}


def padded_layer(self, hidden, layer, state, position):
    for _ in range(EXTRA["n"]):
        ttnn.multiply(EXTRA["t"], EXTRA["t"])
    return real_layer(self, hidden, layer, state, position)


model_mod.TTModel._layer = padded_layer

# The same question for a right-sized `generic_op`, which is what 469 of the
# step's calls were replaced by. In isolation one of those on a single core is
# 1.74 us against a ttnn op's 5.8; the replacement was worth nothing, and this
# is where that has to show up if the isolated figure means anything.
import ttrunner_qwen38_flash_next.tt.ops as ops_mod                  # noqa: E402

small_out = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
kern_prog = ops_mod._ew_program(pad, pad, small_out, ops_mod.EW_MUL, 0, 0, 1)


def padded_kernel_layer(self, hidden, layer, state, position):
    for _ in range(EXTRA["n"]):
        ttnn.generic_op([pad, pad, small_out], kern_prog)
    return real_layer(self, hidden, layer, state, position)


print("RESULT extra ops a layer -> ms a step, and the slope in us an op",
      flush=True)
for label, tensor in (("1 tile", pad), ("320 tiles", WIDE),
                      ("1-core kernel", None)):
    if tensor is None:
        model_mod.TTModel._layer = padded_kernel_layer
    else:
        model_mod.TTModel._layer = padded_layer
        EXTRA["t"] = tensor
    rows = []
    for k in (0, 4, 8):
        EXTRA["n"] = k
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
        rows.append((k, best))
        print(f"RESULT   {label:10s} +{k:2d}/layer ({k * cfg.num_layers:4d} ops): "
              f"{best:7.2f} ms", flush=True)
        dec.release()
    (k0, t0_), (k1, t1_) = rows[0], rows[-1]
    slope = (t1_ - t0_) / ((k1 - k0) * cfg.num_layers) * 1000
    print(f"RESULT   {label:10s} -> {slope:5.2f} us an op "
          f"(the isolated figure is 5.8)", flush=True)

model_mod.TTModel._layer = real_layer
ttnn.close_mesh_device(mesh)
