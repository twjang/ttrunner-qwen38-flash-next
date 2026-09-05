"""Which `ttnn.linear` shapes hold the 27 ms? Counted and timed, per shape.

    uv run python scripts/dev/linear_shape_census.py

The step splits into two halves (handoff 17.1): 5413 small ops at 5.8 us apiece
is 31.4 ms, and 484 linears whose roofline is 14.6 us each measure about 56 --
26 % of bandwidth, 27 ms. The op census groups by op *name*, which is enough for
the first half and useless for the second: what decides a matmul's efficiency is
its shape (invariant 38, output width sets how many cores get work).

So this records every (K, N, dtype) a decode step issues, then times each shape
once in isolation and ranks them by `calls x measured`. That is what says which
three or four shapes to attack rather than all of them.
"""
import sys
import time
from collections import Counter, defaultdict

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

mesh, cfg, m = open_model(max_seq_len=4096)
from ttrunner_qwen38_flash_next.tt.ops import fast_linear, HIFI4      # noqa: E402
m.selection_active = False

shapes = Counter()
real = ttnn.linear


def spy(a, b, *args, **kwargs):
    shapes[(tuple(a.shape), tuple(b.shape), str(a.dtype), str(b.dtype))] += 1
    return real(a, b, *args, **kwargs)


ttnn.linear = spy
state = m.new_state(batch=1)
m.step([1000], state)          # warm
shapes.clear()
m.step([1000], state)
ttnn.linear = real
ttnn.synchronize_device(mesh)

print(f"RESULT {sum(shapes.values())} ttnn.linear calls over "
      f"{len(shapes)} distinct shapes", flush=True)

rep = ttnn.ReplicateTensorToMesh(mesh)
DT = {"DataType.BFLOAT16": ttnn.bfloat16, "DataType.BFLOAT8_B": ttnn.bfloat8_b,
      "DataType.BFLOAT4_B": ttnn.bfloat4_b, "DataType.FLOAT32": ttnn.float32}
BYTES = {"DataType.BFLOAT16": 2, "DataType.BFLOAT8_B": 1.0625,
         "DataType.BFLOAT4_B": 0.5625, "DataType.FLOAT32": 4}


def timed(fn, reps=20, iters=6):
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


rows = []
for (ash, bsh, adt, bdt), n in shapes.items():
    if len(ash) != 4 or len(bsh) != 4:
        continue
    try:
        a = ttnn.from_torch(torch.randn(*ash) * 0.05, dtype=DT[adt],
                            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        b = ttnn.from_torch(torch.randn(*bsh) * 0.02, dtype=DT[bdt],
                            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        us = timed(lambda a=a, b=b: fast_linear(a, b, compute_kernel_config=HIFI4))
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    except Exception as exc:                                          # noqa: BLE001
        print(f"RESULT skipped {ash} x {bsh}: {type(exc).__name__}", flush=True)
        continue
    wbytes = 1
    for d in bsh:
        wbytes *= d
    wbytes *= BYTES[bdt]
    roof = wbytes / 388e9 * 1e6
    rows.append((n * us / 1000.0, n, us, roof, ash, bsh, bdt))

rows.sort(reverse=True)
print(f"RESULT {'ms/token':>9s} {'calls':>6s} {'us':>8s} {'roof us':>8s} "
      f"{'eff':>6s}  shape", flush=True)
total = 0.0
for ms, n, us, roof, ash, bsh, bdt in rows:
    total += ms
    eff = roof / us * 100 if us else 0
    print(f"RESULT {ms:9.2f} {n:6d} {us:8.2f} {roof:8.2f} {eff:5.1f}%  "
          f"[{ash[-2]},{ash[-1]}] x [{bsh[-2]},{bsh[-1]}] {bdt.split('.')[-1]}",
          flush=True)
print(f"RESULT ---", flush=True)
print(f"RESULT {total:.2f} ms a token in ttnn.linear "
      f"(through `fast_linear`, i.e. what the model pays)", flush=True)
roof = sum(r[3] * r[1] for r in rows) / 1000.0
print(f"RESULT roofline for the same weights: {roof:.2f} ms "
      f"-> {roof / max(total, 1e-9) * 100:.0f}% of bandwidth", flush=True)

ttnn.close_mesh_device(mesh)
