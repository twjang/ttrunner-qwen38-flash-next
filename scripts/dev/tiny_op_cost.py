"""What a tiny op really costs inside a trace, at the shapes decode uses.

    uv run python scripts/dev/tiny_op_cost.py

The op census says one decode step issues 7064 ttnn calls, and that 3959 of them
-- multiply 1297, reshape 937, slice 588, add 581, sigmoid 326, silu 230 -- move
0.28 GB between them. They are glue, not work.

Whether that matters depends entirely on the per-call cost, and the answer
changes the plan:

  ~1.4 us  (the traced dispatch floor measured earlier)  -> 5.5 ms, ignore them
  ~10 us                                                  -> 40 ms, fusion is
                                                             the whole game

So measure it, at the widths the residual stream actually carries (2560 and the
hc_count=4 expansion at 10240) and on the expert intermediates the MoE's SwiGLU
chain works over.

**At the row count decode actually runs**, which is 1, not 32. An earlier version
of this script used 32 and its numbers were wrong for the thing being decided:
it priced the `gated_residual_mix` reshape at 92.3 us and predicted 5.3 ms from
replacing it, where the real saving measured 0.23. A row is tile-padded to 32 in
*storage*, but the op still walks only the tile-rows the logical shape has, so
M=1 and M=32 are not the same call.

Timed inside a trace with many copies per capture, so what comes out is the
marginal cost of one more op and not a dispatch round trip.
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                # noqa: E402

REPS = 200        # ops per trace: the marginal cost is what matters
ITERS = 10

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)


def dev(*shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(*shape) * 0.1, dtype=dtype,
                           layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)


def marginal(fn, label):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(REPS):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    us = best / REPS * 1e6
    print(f"RESULT {label:46s} {us:8.2f} us/call", flush=True)
    return us


# --- the residual stream, 2560 and the 4x hyper-connection expansion --------
M = 1        # what decode runs
a = dev(1, 1, M, 2560)
b = dev(1, 1, M, 2560)
w = dev(1, 1, M, 10240)
v = dev(1, 1, M, 10240)

costs = {}
costs["mul 32x2560"] = marginal(lambda: ttnn.multiply(a, b), f"multiply [1,1,{M},2560]")
costs["add 32x2560"] = marginal(lambda: ttnn.add(a, b), f"add      [1,1,{M},2560]")
costs["silu 32x2560"] = marginal(lambda: ttnn.silu(a), f"silu     [1,1,{M},2560]")
costs["sig 32x2560"] = marginal(lambda: ttnn.sigmoid(a), f"sigmoid  [1,1,{M},2560]")
costs["mul 32x10240"] = marginal(lambda: ttnn.multiply(w, v), f"multiply [1,1,{M},10240]")
costs["resh 32x10240"] = marginal(
    lambda: ttnn.reshape(w, (1, M, 4, 2560)), f"reshape  [1,1,{M},10240]->[1,{M},4,2560]")
costs["slice 32x10240"] = marginal(
    lambda: ttnn.slice(w, (0, 0, 0, 0), (1, 1, M, 2560)), f"slice    [1,1,{M},10240]->2560")

# --- the expert intermediates the MoE SwiGLU chain works over ---------------
E = 128
big = dev(1, E, M, 1280)
big2 = dev(1, E, M, 1280)
costs["mul E128x1280"] = marginal(lambda: ttnn.multiply(big, big2),
                                  f"multiply [1,{E},{M},1280]  (expert interm.)")
costs["slice E128"] = marginal(
    lambda: ttnn.slice(big, (0, 0, 0, 0), (1, E, M, 640)),
    f"slice    [1,{E},{M},1280]->640")
costs["silu E128"] = marginal(lambda: ttnn.silu(big), f"silu     [1,{E},{M},1280]")

print("RESULT ---", flush=True)
glue = 3959
small = min(costs[k] for k in ("mul 32x2560", "add 32x2560", "sig 32x2560"))
print(f"RESULT cheapest small op is {small:.2f} us -- so 3959 glue calls is at "
      f"least {glue * small / 1e3:.1f} ms of the 104 ms step", flush=True)
print(f"RESULT the expert-wide ops are the expensive ones: multiply over "
      f"[1,128,32,1280] is {costs['mul E128x1280']:.1f} us, and the SwiGLU chain "
      f"runs 3 of those a layer", flush=True)
per_layer = costs["mul E128x1280"] + costs["slice E128"] * 2 + costs["silu E128"]
print(f"RESULT MoE SwiGLU chain: {per_layer:.1f} us/layer x 48 = "
      f"{per_layer * 48 / 1e3:.2f} ms a token, on tensors that are 97 % zeros",
      flush=True)

ttnn.close_mesh_device(mesh)
