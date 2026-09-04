"""A fused SwiGLU kernel for the MoE, against the four ttnn ops it replaces.

    uv run python scripts/stage3_swiglu_fusion.py

`expert_ffn` turns the fused gate|up matmul output `[1, E, M, 2N]` into
`silu(gate) * up` with four ttnn ops -- two slices, a silu, a multiply. Measured
as a unit that is **142 us a layer, 6.83 ms a token**.

It is not op overhead. The four ops move ~61 MB a layer between them, because
each reads and writes the whole E=128 tensor, and 61 MB at 388 GB/s is 157 us --
which is the measured 142. So it is already at bandwidth, and the only way to
make it faster is to move less: one fused pass reads both halves once and writes
the result once, 16.7 MB, ~43 us.

`ttnn.swiglu` does not help -- it is a composite that issues the same four ops
(invariant 43). This is the real thing: reader, compute and writer kernels under
`ttnn.generic_op`, with the intermediate living in circular buffers and never
reaching DRAM.

Correctness first: the fused result is compared against the ttnn chain on the
same input before anything is timed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import ttnn

KDIR = Path(__file__).resolve().parent / "kernels"
E, M, N = 128, 1, 640          # this device's experts, decode's one row, half-width
READ_BATCH_CB = 4              # tiles of slack in each circular buffer


def core_list(grid):
    return [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]


def split_work(total: int, n: int):
    return [((total * c) // n, (total * (c + 1)) // n) for c in range(n)]


def build(src, out, grid):
    cores = core_list(grid)
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )
    acc = {}
    for tag, t in (("i", src), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"{tag}: expected an interleaved tensor, got {len(ct)} CT args")
        acc[tag] = ct
    tile_bytes = acc["i"][1]
    if acc["o"][1] != tile_bytes:
        raise RuntimeError("input and output pages differ; the kernels move whole tiles")

    nt_out = out.shape[-1] // 32
    rows = out.shape[1] * (out.shape[2] // 32 if out.shape[2] >= 32 else 1)
    total_work = rows * nt_out

    cbs = [
        ttnn.CBDescriptor(
            total_size=READ_BATCH_CB * tile_bytes, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=i, data_format=src.dtype, page_size=tile_bytes)])
        for i in (0, 1, 2)
    ]

    work = split_work(total_work, len(cores))
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "swiglu_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=[nt_out, tile_bytes] + acc["i"],
        runtime_args=[(c, [src.buffer_address(), lo, hi]) for c, (lo, hi) in zip(cores, work)],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "swiglu_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=[],
        runtime_args=[(c, [hi - lo]) for c, (lo, hi) in zip(cores, work)],
        config=ttnn.ComputeConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "swiglu_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=[tile_bytes] + acc["o"],
        runtime_args=[(c, [out.buffer_address(), lo, hi]) for c, (lo, hi) in zip(cores, work)],
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=cbs), total_work


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dev"))
    from _device_model import open_model                              # noqa: E402

    mesh, cfg, m = open_model(max_seq_len=512)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    torch.manual_seed(0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        host = torch.randn(1, E, M, 2 * N) * 0.5
        src = ttnn.from_torch(host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                              device=mesh, mesh_mapper=rep)
        out = ttnn.from_torch(torch.zeros(1, E, M, N), dtype=ttnn.bfloat8_b,
                              layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        prog, total_work = build(src, out, grid)
        print(f"RESULT src {tuple(src.shape)} {src.dtype}, {total_work} output tiles, "
              f"grid {grid.x}x{grid.y}", flush=True)

        def chain():
            g = ttnn.slice(src, (0, 0, 0, 0), (1, E, M, N))
            u = ttnn.slice(src, (0, 0, 0, N), (1, E, M, 2 * N))
            return ttnn.multiply(ttnn.silu(g), u)

        # --- correctness before speed --------------------------------------
        ttnn.generic_op([src, out], prog)
        ttnn.synchronize_device(mesh)
        got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
        ref = ttnn.to_torch(chain(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
        d = (got.to(torch.float64) - ref.to(torch.float64)).abs()
        scale = ref.to(torch.float64).abs().max().item()
        print(f"RESULT fused vs ttnn chain: max abs err {d.max().item():.4e} "
              f"(values to {scale:.3e}, rel {d.max().item() / max(scale, 1e-30):.2e})",
              flush=True)
        ok = d.max().item() <= 0.02 * scale
        print(f"RESULT correctness: {'OK' if ok else 'MISMATCH -- not timing a wrong answer'}",
              flush=True)
        if not ok:
            raise SystemExit(1)

        def timed(fn, label, reps=20, iters=8):
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
            us = best / reps * 1e6
            print(f"RESULT {label:40s} {us:8.2f} us  -> {48 * us / 1e3:6.2f} ms/48 layers",
                  flush=True)
            return us

        a = timed(chain, "slice+slice+silu+multiply (today)")
        b = timed(lambda: ttnn.generic_op([src, out], prog), "fused SwiGLU kernel")
        print("RESULT ---", flush=True)
        print(f"RESULT {a:.1f} -> {b:.1f} us a layer ({a / b:.2f}x), "
              f"{48 * (a - b) / 1e3:+.2f} ms a token", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
