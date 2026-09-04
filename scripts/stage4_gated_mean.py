"""A fused gate-and-average kernel for the hyper-connection block.

    uv run python scripts/stage4_gated_mean.py

`gated_residual_mix` ends with nine ttnn ops: `multiply(mix, normed)` over a
10240-wide pair, four tile-aligned slices to pull out the hc streams, three adds
and a scale. Two costs, and both point the same way.

Bytes: those nine round-trip about 11 MB a call between them. Fused, one output
tile reads eight input tiles and writes one -- 1.3 MB read, 164 KB written.

Per-op floor: a ttnn op costs ~5.5 us whatever it touches (invariant 42), so
nine of them is ~50 us before any data moves, against a `generic_op` launch at
roughly 12 us. Fusing pays here precisely because nine is well past the three
that would break even.

Correctness is checked against the ttnn chain before anything is timed.
"""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import torch
import ttnn

KDIR = Path(__file__).resolve().parent / "kernels"
HC, HIDDEN, M = 4, 2560, 1


def core_list(grid):
    return [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]


def build(a, b, out, grid, hc: int):
    cores = core_list(grid)
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )
    acc = {}
    for tag, t in (("a", a), ("b", b), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"{tag}: expected an interleaved tensor")
        acc[tag] = ct
    tile_bytes = acc["a"][1]

    nt_out = out.shape[-1] // 32
    rows = max(out.shape[2] // 32, 1) * out.shape[1]
    total = rows * nt_out
    work = [((total * c) // len(cores), (total * (c + 1)) // len(cores))
            for c in range(len(cores))]

    cbs = [
        ttnn.CBDescriptor(
            total_size=(hc + 1) * tile_bytes, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=i, data_format=a.dtype, page_size=tile_bytes)])
        for i in (0, 1)
    ] + [
        ttnn.CBDescriptor(
            total_size=2 * tile_bytes, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=2, data_format=out.dtype, page_size=tile_bytes)])
    ]

    inv_bits = struct.unpack("<I", struct.pack("<f", 1.0 / hc))[0]

    def kern(name, ct, args, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=str(KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfg)

    return ttnn.ProgramDescriptor(
        kernels=[
            kern("gated_mean_reader.cpp", [nt_out, hc, tile_bytes] + acc["a"] + acc["b"],
                 [[a.buffer_address(), b.buffer_address(), lo, hi] for lo, hi in work],
                 ttnn.ReaderConfigDescriptor()),
            kern("gated_mean_compute.cpp", [hc, inv_bits],
                 [[hi - lo] for lo, hi in work], ttnn.ComputeConfigDescriptor()),
            kern("gated_mean_writer.cpp", [tile_bytes] + acc["o"],
                 [[out.buffer_address(), lo, hi] for lo, hi in work],
                 ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[], cbs=cbs), total


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dev"))
    from _device_model import open_model                              # noqa: E402

    mesh, cfg, m = open_model(max_seq_len=512)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    torch.manual_seed(0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        wide = HC * HIDDEN
        ha = torch.randn(1, 1, M, wide) * 0.4
        hb = torch.randn(1, 1, M, wide) * 0.4
        a = ttnn.from_torch(ha, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh, mesh_mapper=rep)
        b = ttnn.from_torch(hb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh, mesh_mapper=rep)
        out = ttnn.from_torch(torch.zeros(1, 1, M, HIDDEN), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        prog, total = build(a, b, out, grid, HC)
        print(f"RESULT {tuple(a.shape)} x2 -> {tuple(out.shape)}, {total} output tiles, "
              f"grid {grid.x}x{grid.y}", flush=True)

        def chain():
            g = ttnn.multiply(a, b)
            parts = [ttnn.slice(g, (0, 0, 0, h * HIDDEN), (1, 1, M, (h + 1) * HIDDEN))
                     for h in range(HC)]
            t = parts[0]
            for p in parts[1:]:
                t = ttnn.add(t, p)
            return ttnn.multiply(t, 1.0 / HC)

        ttnn.generic_op([a, b, out], prog)
        ttnn.synchronize_device(mesh)
        got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
        ref = ttnn.to_torch(chain(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
        d = (got.to(torch.float64) - ref.to(torch.float64)).abs()
        scale = ref.to(torch.float64).abs().max().item()
        rel = d.max().item() / max(scale, 1e-30)
        print(f"RESULT fused vs ttnn chain: max abs err {d.max().item():.4e} "
              f"(values to {scale:.3e}, rel {rel:.2e})", flush=True)
        ok = rel <= 0.02
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
            print(f"RESULT {label:40s} {us:8.2f} us  -> {97 * us / 1e3:6.2f} ms/97 calls",
                  flush=True)
            return us

        x = timed(chain, "multiply+4 slices+3 adds+scale")
        y = timed(lambda: ttnn.generic_op([a, b, out], prog), "fused gated-mean kernel")
        print("RESULT ---", flush=True)
        print(f"RESULT {x:.1f} -> {y:.1f} us a call ({x / y:.2f}x), "
              f"{97 * (x - y) / 1e3:+.2f} ms a token", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
