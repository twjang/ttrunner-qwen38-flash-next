"""Does a launch cost by its core count, or by being a *different* program?

    uv run python scripts/dev/program_switch_cost.py

`dispatch_floor.py` priced a `generic_op` at 2.06 us on one core and 5.81 on a
hundred and ten, and concluded that ttnn's full-grid launches waste ~10 ms of the
step. Acting on it -- 613 of the step's ops replaced by right-sized kernels,
confirmed by the census dropping 5440 -> 4827 calls -- returned **0.28 ms**, not
the 2.27 that 613 x 3.7 us predicts.

The one thing the probe did that the model does not is repeat the *same* program.
A trace of sixty-four identical launches may be measuring something a trace of
sixty-four different ones does not.

So: N copies of one program against N distinct programs, same work each, at one
core and at a hundred and ten.
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
N = 32


def timed(fn, reps=8, iters=8):
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
    return best / (reps * N) * 1e6


a = ttnn.from_torch(torch.randn(1, 1, 32, 320), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
b = ttnn.from_torch(torch.zeros(1, 1, 32, 320), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
acc = list(ttnn.TensorAccessorArgs(a).get_compile_time_args())


def nop_program(n_cores, tag):
    cols = min(n_cores, grid.x)
    rows = (n_cores + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(x, y) for y in range(rows) for x in range(cols)]
    cb = ttnn.CBDescriptor(
        total_size=2 * acc[1], core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=0, data_format=a.dtype, page_size=acc[1])])
    # `tag` goes in the compile-time args, which is what makes each of these a
    # *distinct* program: generic_op hashes those by value (handoff 4g).
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/nop_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs, compile_time_args=acc,
        runtime_args=[(c, [a.buffer_address()]) for c in cores],
        config=ttnn.ReaderConfigDescriptor())
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])


# How the cost behaves as the trace gets long, because the model's trace is
# ~4800 launches and this probe's was 256. If the dispatcher rather than the
# cores is the limit, the one-core advantage should fade.
print(f"RESULT {'cores':>6s} {'launches':>9s} {'us a launch':>12s}", flush=True)
for n_cores in (1, 8, 110):
    prog = nop_program(n_cores, 0)
    for reps in (8, 64, 256):
        us = timed(lambda p=prog: [ttnn.generic_op([a, b], p) for _ in range(N)],
                   reps=reps, iters=4)
        print(f"RESULT {n_cores:6d} {reps * N:9d} {us:11.2f}us", flush=True)

print("RESULT ---", flush=True)
print(f"RESULT {N} launches a group", flush=True)
print(f"RESULT {'cores':>6s} {'same program':>14s} {'distinct programs':>18s}",
      flush=True)
for n_cores in (1, 8, 110):
    one = nop_program(n_cores, 0)
    many = [nop_program(n_cores, i) for i in range(N)]
    # distinct *programs* need distinct descriptors; ttnn caches on the
    # descriptor, so a fresh object per launch is what the model does anyway.
    t_same = timed(lambda p=one: [ttnn.generic_op([a, b], p) for _ in range(N)])
    t_diff = timed(lambda ps=many: [ttnn.generic_op([a, b], p) for p in ps])
    print(f"RESULT {n_cores:6d} {t_same:12.2f}us {t_diff:16.2f}us", flush=True)

# and the same question for ttnn's own ops: one op repeated, against a mix
tensors = [ttnn.from_torch(torch.randn(1, 1, 32, 320), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
           for _ in range(N)]
t_same = timed(lambda: [ttnn.multiply(a, a) for _ in range(N)])
mix = [ttnn.multiply, ttnn.add, ttnn.subtract, ttnn.multiply]
t_diff = timed(lambda: [mix[i % 4](tensors[i], tensors[i]) for i in range(N)])
print(f"RESULT ttnn: one op repeated {t_same:6.2f}us, four ops mixed {t_diff:6.2f}us",
      flush=True)

ttnn.close_mesh_device(mesh)
