"""A right-sized elementwise kernel against `ttnn.multiply`, by tile count.

    uv run python scripts/dev/small_ew_probe.py

`dispatch_floor.py`: a launch costs about 1.8 us plus 0.036 us a core, so one
core is 2.06 and a hundred and ten is 5.81. ttnn's elementwise ops always take
the whole grid, and `step_op_census.py` says 3150 of the step's 5764 calls touch
64 tiles or fewer -- about 10 ms of launching cores with nothing to do.

This prices the alternative: the same arithmetic in a `generic_op` whose core
range is `min(tiles, grid)`. Correctness first, then the clock.
"""
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device_model import open_model                                 # noqa: E402

KDIR = Path(__file__).resolve().parents[1] / "kernels"

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
torch.manual_seed(0)
grid = mesh.compute_with_storage_grid_size()
N_CORES = grid.x * grid.y


def build(a, b, out, op, scalar=0):
    n_tiles = 1
    for d in list(out.shape)[:-2]:
        n_tiles *= d
    n_tiles *= max(1, (out.shape[-2] + 31) // 32) * max(1, (out.shape[-1] + 31) // 32)
    n = min(n_tiles, N_CORES)
    cols = min(n, grid.x)
    rows = (n + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(x, y) for y in range(rows) for x in range(cols)]
    binary = op <= 2
    acc = {t: list(ttnn.TensorAccessorArgs(v).get_compile_time_args())
           for t, v in (("a", a), ("b", b if binary else a), ("o", out))}
    tb = acc["a"][1]
    work = [((n_tiles * c) // len(cores), (n_tiles * (c + 1)) // len(cores))
            for c in range(len(cores))]
    cbs = [ttnn.CBDescriptor(
        total_size=2 * tb, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=a.dtype, page_size=tb)]) for i in (0, 1, 2)]

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    return ttnn.ProgramDescriptor(kernels=[
        kern("small_ew_reader.cpp", [int(binary), tb] + acc["a"] + acc["b"],
             [[a.buffer_address(), (b if binary else a).buffer_address(), lo, hi]
              for lo, hi in work], ttnn.ReaderConfigDescriptor()),
        kern("small_ew_compute.cpp", [op, scalar], [[hi - lo] for lo, hi in work],
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("small_ew_writer.cpp", acc["o"],
             [[out.buffer_address(), lo, hi] for lo, hi in work],
             ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs), n


def timed(fn, reps=48, iters=8):
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


comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
print(f"RESULT {'shape':22s} {'tiles':>6s} {'cores':>6s} {'ttnn':>8s} "
      f"{'kernel':>8s} {'':>6s} {'exact?':>7s}", flush=True)
for shape in [(1, 1, 32, 32), (1, 1, 32, 128), (1, 1, 32, 512), (1, 1, 32, 2048),
              (1, 1, 32, 10240)]:
    a = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    b = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    out = ttnn.from_torch(torch.zeros(*shape), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    prog, n = build(a, b, out, 0)
    ttnn.generic_op([a, b, out], prog)
    ttnn.synchronize_device(mesh)
    got = ttnn.to_torch(out, mesh_composer=comp)[:1]
    ref = ttnn.to_torch(ttnn.multiply(a, b), mesh_composer=comp)[:1]
    # against float64 on the operands the device holds, because differing from
    # ttnn is not the same as being wrong (invariant 57)
    a64 = ttnn.to_torch(a, mesh_composer=comp)[:1].to(torch.float64)
    b64 = ttnn.to_torch(b, mesh_composer=comp)[:1].to(torch.float64)
    truth = a64 * b64
    sc = truth.abs().max().item()
    e_k = (got.to(torch.float64) - truth).abs().max().item() / sc
    e_t = (ref.to(torch.float64) - truth).abs().max().item() / sc
    d = (got - ref).abs().max().item()
    t_ttnn = timed(lambda a=a, b=b: ttnn.multiply(a, b))
    t_kern = timed(lambda a=a, b=b, out=out, p=prog: ttnn.generic_op([a, b, out], p))
    tiles = (shape[2] // 32) * (shape[3] // 32)
    print(f"RESULT {str(shape):22s} {tiles:6d} {n:6d} {t_ttnn:7.2f}u "
          f"{t_kern:7.2f}u {t_ttnn/t_kern:5.2f}x  vs float64: kernel {e_k:.2e} "
          f"ttnn {e_t:.2e}", flush=True)
    for t in (a, b, out):
        ttnn.deallocate(t)

ttnn.close_mesh_device(mesh)
