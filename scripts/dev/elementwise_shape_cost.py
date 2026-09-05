"""Is a traced elementwise op paid for by its shape, or by a fixed launch?

    uv run python scripts/dev/elementwise_shape_cost.py

Invariant 42 says an elementwise op costs ~5.5 us whatever its shape, and the
whole plan for reaching 32.6 ms has been built on that: 5897 calls x 5.5 us is
32 ms of the 58 ms step, so fuse ops away.

Two measurements since then do not fit. The census's own arithmetic uses a 1.4 us
traced dispatch floor, and the fused `reinject` removed 288 calls for 0.40 ms --
which is 1.39 us a call, not 5.5. If the floor really is ~1.4 us then op count is
8 ms of the step and the other 50 is **bytes**, and the biggest source of bytes
at M=1 is padding: a [1, 1, 1, 10240] bfloat16 tensor is 320 tiles, 640 KB, of
which 20 KB is real. Every elementwise op reads and writes all of it.

This settles which it is. Same element count, different padding:

    [1, 1, 1, 10240]   320 tiles   640 KB   <- one real row in thirty-two
    [1, 1, 32, 320]     10 tiles    20 KB   <- the same 10240 elements, packed
    [1, 1, 1, 320]       10 tiles   20 KB
    [1, 1, 1, 32]         1 tile     2 KB

If the cost is a launch, all four are the same. If it is bytes, the first is 32x
the second -- and then the lever is the layout, not the op count.
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

SHAPES = [
    ("[1,1,1,10240]  320 tiles", (1, 1, 1, 10240)),
    ("[1,1,32,320]    10 tiles", (1, 1, 32, 320)),
    ("[1,1,1,320]     10 tiles", (1, 1, 1, 320)),
    ("[1,1,32,10240] 320 tiles", (1, 1, 32, 10240)),
    ("[1,1,1,32]       1 tile ", (1, 1, 1, 32)),
    ("[1,1,1,2560]    80 tiles", (1, 1, 1, 2560)),
]


def timed(fn, reps=60, iters=8):
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


print(f"RESULT {'shape':28s} {'multiply':>10s} {'sigmoid':>10s} "
      f"{'phys KB':>9s} {'GB/s':>8s}", flush=True)
for label, shape in SHAPES:
    a = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    b = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    tiles = ((shape[2] + 31) // 32) * ((shape[3] + 31) // 32) * shape[0] * shape[1]
    phys = tiles * 2048
    mul = timed(lambda a=a, b=b: ttnn.multiply(a, b))
    sig = timed(lambda a=a: ttnn.sigmoid(a))
    # a multiply reads two operands and writes one
    bw = 3 * phys / (mul * 1e-6) / 1e9
    print(f"RESULT {label:28s} {mul:9.2f}u {sig:9.2f}u {phys/1024:9.0f} {bw:8.1f}",
          flush=True)

# --- and the other op *kinds*, which the multiply/sigmoid pair does not cover.
#
# Two fused kernels have now returned about a tenth of what "5.8 us x ops
# removed" predicted, and both removed slices, reshapes and permutes rather than
# multiplies. If those are cheap the whole fusion backlog is worth far less than
# it looks.
print("RESULT ---", flush=True)
print(f"RESULT {'op':16s} " + " ".join(f"{l.split()[0]:>16s}" for l, _ in SHAPES[:4]),
      flush=True)
tensors = {sh: ttnn.from_torch(torch.randn(*sh), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
           for _, sh in SHAPES[:4]}


def row(name, make):
    cells = []
    for _, sh in SHAPES[:4]:
        t = tensors[sh]
        try:
            fn = make(t, sh)
            cells.append(f"{timed(fn):13.2f}us" if fn is not None else f"{'n/a':>16s}")
        except Exception as exc:                                      # noqa: BLE001
            cells.append(f"{type(exc).__name__:>16s}")
    print(f"RESULT {name:16s} " + " ".join(cells), flush=True)


row("multiply", lambda t, sh: (lambda: ttnn.multiply(t, t)))
row("sigmoid", lambda t, sh: (lambda: ttnn.sigmoid(t)))
row("slice half", lambda t, sh: (lambda: ttnn.slice(
    t, (0, 0, 0, 0), (sh[0], sh[1], sh[2], max(32, sh[3] // 2)))))
row("slice 4 cols", lambda t, sh: (lambda: ttnn.slice(t, (0, 0, 0, 0), (sh[0], sh[1], sh[2], 32))))
row("permute", lambda t, sh: (lambda: ttnn.permute(t, (0, 1, 3, 2))))
row("reshape", lambda t, sh: (
    (lambda: ttnn.reshape(t, (sh[0], sh[1], sh[2] * 4, sh[3] // 4)))
    if sh[3] % 128 == 0 else None))
row("concat self", lambda t, sh: (lambda: ttnn.concat([t, t], dim=-1)))
row("sum -1", lambda t, sh: (lambda: ttnn.sum(t, dim=-1, keepdim=True)))
row("typecast", lambda t, sh: (lambda: ttnn.typecast(t, ttnn.float32)))
row("add", lambda t, sh: (lambda: ttnn.add(t, t)))

ttnn.close_mesh_device(mesh)
