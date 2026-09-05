"""What does a launch cost, as opposed to an op?

    uv run python scripts/dev/dispatch_floor.py

Every plan for reaching 32.6 ms a token at batch 1 runs into the same number:
5764 ttnn calls at 5.8 us apiece is 33 ms before any arithmetic, and an op costs
that whatever its shape (invariant 66). But `step_op_census.py` carries a
different figure in its own arithmetic -- "the 1.4 us traced dispatch floor" --
and the two cannot both be right.

If a `generic_op` that touches one page costs 1.4 us where `ttnn.multiply` costs
5.8, then four fifths of the step's floor is what ttnn's ops spend *above*
dispatch, and the lever is not fusing them but replacing them.

Also varies the core count, because a launch that has to reach 110 cores and
collect 110 acknowledgements might not cost what a launch to one core does.
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

KDIR = __file__.rsplit("/", 2)[0] + "/kernels"

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
torch.manual_seed(0)
grid = mesh.compute_with_storage_grid_size()


def timed(fn, reps=64, iters=8):
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


a = ttnn.from_torch(torch.randn(1, 1, 32, 320), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
b = ttnn.from_torch(torch.zeros(1, 1, 32, 320), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
acc = list(ttnn.TensorAccessorArgs(a).get_compile_time_args())
page = acc[1]


def nop_program(n_cores):
    cols = min(n_cores, grid.x)
    rows = (n_cores + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(x, y) for y in range(rows) for x in range(cols)]
    cb = ttnn.CBDescriptor(
        total_size=2 * page, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=0, data_format=a.dtype, page_size=page)])
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/nop_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs, compile_time_args=acc,
        runtime_args=[(c, [a.buffer_address()]) for c in cores],
        config=ttnn.ReaderConfigDescriptor())
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])


print(f"RESULT ttnn.multiply [1,1,32,320]   {timed(lambda: ttnn.multiply(a, a)):7.2f} us",
      flush=True)
print(f"RESULT ttnn.sigmoid  [1,1,32,320]   {timed(lambda: ttnn.sigmoid(a)):7.2f} us",
      flush=True)
for n in (1, 8, 32, 64, 110):
    try:
        prog = nop_program(n)
        us = timed(lambda p=prog: ttnn.generic_op([a, b], p))
        print(f"RESULT generic_op, one page, {n:3d} cores {us:7.2f} us", flush=True)
    except Exception as exc:                                          # noqa: BLE001
        print(f"RESULT generic_op {n:3d} cores: {type(exc).__name__}: "
              f"{(str(exc) or repr(exc)).splitlines()[0][:100]}", flush=True)

ttnn.close_mesh_device(mesh)
