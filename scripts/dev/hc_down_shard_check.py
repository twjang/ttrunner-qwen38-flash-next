"""Would K-sharding `hc_down` across the four devices pay for its all_reduce?

    uv run python scripts/dev/hc_down_shard_check.py

`hc_down` is [10240, 640], **replicated**, and runs 96 times a token (twice a
layer inside `gated_residual_mix`). It reads 3.4 MB per call at 26.6 GB/s -- 7 %
of this box's bandwidth -- because N=640 is 20 output tiles and cannot fill the
grid. Ablating it costs 18.77 ms of a 146.2 ms step, and DRAM sharding is
refused for this shape (handoff 5.1), so the stock-config routes are exhausted.

Splitting K four ways is the one structural fix that needs no kernel: each device
holds [2560, 640], reads a quarter of the bytes, and an all_reduce over the
640-wide partial puts the full result back on every device. It is
precision-neutral -- the same products, summed in a different order, and a tree
reduction is if anything better conditioned than one core's serial K loop.

The question is entirely whether 96 extra all_reduces cost less than the 3/4 of
13.3 ms they would save. The 3.77 ms measured for the model's 84 collectives is
an average over 2560- and 10240-wide tensors, so it says nothing about a 640-wide
one; that is what this measures.

Everything is timed inside a trace, since that is where it would run.
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                # noqa: E402

M = 32            # decode's single row, tile-padded, which is what it costs
REPS = 20
ITERS = 10

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)


def dev(t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                           mesh_mapper=rep)


def traced(fn, label):
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
    per = best / REPS * 1e3
    print(f"RESULT {label:42s} {per:8.4f} ms/call", flush=True)
    return per


# --- the collective, at the widths that matter -------------------------------
for width in (640, 2560, 10240):
    t = dev(torch.randn(1, 1, M, width) * 0.1)
    traced(lambda t=t: ttnn.all_reduce(t, cluster_axis=1, topology=ttnn.Topology.Linear),
           f"all_reduce {M}x{width}")
    ttnn.deallocate(t)

# --- the matmul, replicated versus K-sharded --------------------------------
# The real tensor, read off the cache manifest rather than guessed: GGUF stores
# hc_down as (out, in) = [320, 10240], so the device layout is [10240, 320],
# dtype bfloat8_b, 3.48 MB. An earlier run of this script used [10240, 640] in
# bfloat4_b -- a wider output and half the bytes -- and its numbers do not apply.
N_DOWN = 320
W_DTYPE = ttnn.bfloat8_b

x_full = dev(torch.randn(1, 1, M, 10240) * 0.05)
w_full = dev(torch.randn(1, 1, 10240, N_DOWN) * 0.02, W_DTYPE)
now = traced(lambda: ttnn.linear(x_full, w_full),
             f"hc_down replicated 10240x{N_DOWN} bf8")

x_part = dev(torch.randn(1, 1, M, 2560) * 0.05)
w_part = dev(torch.randn(1, 1, 2560, N_DOWN) * 0.02, W_DTYPE)
shard = traced(lambda: ttnn.linear(x_part, w_part),
               f"hc_down K-sharded  2560x{N_DOWN} bf8")

# The other two replicated readers in the same block, at their real shapes.
x_up = dev(torch.randn(1, 1, M, N_DOWN) * 0.05)
w_up = dev(torch.randn(1, 1, N_DOWN, 10240) * 0.02, W_DTYPE)
traced(lambda: ttnn.linear(x_up, w_up), f"hc_up   replicated {N_DOWN}x10240 bf8")

x_r = dev(torch.randn(1, 1, M, 2560) * 0.05)
w_r = dev(torch.randn(1, 1, 2560, 512) * 0.02, ttnn.float32)
traced(lambda: ttnn.linear(x_r, w_r), "router  replicated 2560x512 fp32")
w_r16 = dev(torch.randn(1, 1, 2560, 512) * 0.02, ttnn.bfloat16)
traced(lambda: ttnn.linear(x_r, w_r16), "router  replicated 2560x512 bf16")


def combined():
    p = ttnn.linear(x_part, w_part)
    return ttnn.all_reduce(p, cluster_axis=1, topology=ttnn.Topology.Linear)


both = traced(combined, "K-sharded matmul + all_reduce")

print("RESULT ---", flush=True)
print(f"RESULT per call: {now:.4f} now vs {both:.4f} sharded+reduced "
      f"({now / both:.2f}x)", flush=True)
print(f"RESULT over 96 calls a token: {96 * now:.2f} ms -> {96 * both:.2f} ms "
      f"({96 * (now - both):+.2f} ms)", flush=True)
print(f"RESULT for scale, ablating hc_down measured 18.77 ms of a 146.2 ms step",
      flush=True)
print("RESULT verdict: " + ("worth converting the hc_down weights to a K shard"
                            if both < now * 0.85 else
                            "not worth it -- the collective eats the saving"),
      flush=True)

ttnn.close_mesh_device(mesh)
