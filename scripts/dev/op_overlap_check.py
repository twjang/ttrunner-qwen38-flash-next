"""Do independent ops overlap inside a captured trace?

    uv run python scripts/dev/op_overlap_check.py

Two fused kernels have now returned about a tenth of what the op-count model
predicted -- `reinject` 0.40 ms where 288 removed calls said 1.6, and the raw
gate stream 0.2 ms where 384 said 1.85. `elementwise_shape_cost.py` says an op
really does cost 5.8 us when timed on its own, so the discrepancy is not in the
per-op number.

Which leaves the assumption underneath it: that a step's cost is the *sum* of
its ops. That holds only if they serialise. The probe that measured 5.8 us ran
sixty copies of the same op on the same tensor -- a dependency chain, forced to
serialise. A real step is full of ops that do not depend on each other.

So: N ops on one tensor (a chain) against N ops on N tensors (independent),
inside a trace, at the same shapes. If the second is cheaper per op then the
device overlaps them, the op floor is not additive, and the remaining fusion
backlog is worth a fraction of what it looks like.
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
torch.manual_seed(0)

N = 32
SHAPES = [("tiny  [1,1,1,32]", (1, 1, 1, 32)),
          ("small [1,1,1,320]", (1, 1, 1, 320)),
          ("wide  [1,1,1,10240]", (1, 1, 1, 10240))]


def timed(fn, reps=12, iters=8):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    return best / reps * 1e6


print(f"RESULT {N} ops a group, in a trace", flush=True)
print(f"RESULT {'shape':22s} {'chained':>12s} {'independent':>14s} {'per op':>18s}",
      flush=True)
for label, shape in SHAPES:
    pool = [ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            for _ in range(N)]
    a = pool[0]

    def chained():
        # each sigmoid consumes the previous one's output: no overlap possible
        t = a
        for _ in range(N):
            t = ttnn.sigmoid(t)
        return t

    def independent():
        # N separate tensors, nothing consumes anything: overlap if the device
        # is willing
        for t in pool:
            ttnn.sigmoid(t)

    c, i = timed(chained), timed(independent)
    print(f"RESULT {label:22s} {c:10.2f}us {i:12.2f}us "
          f"{c/N:7.2f} vs {i/N:6.2f}us", flush=True)
    for t in pool:
        ttnn.deallocate(t)

ttnn.close_mesh_device(mesh)
