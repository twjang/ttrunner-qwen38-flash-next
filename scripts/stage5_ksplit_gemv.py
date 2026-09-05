"""A GEMV that splits its reduction across cores, against `ttnn.linear`.

    uv run python scripts/stage5_ksplit_gemv.py

`ttnn.linear` at M=1 runs at 25-33 % of bandwidth whenever its output is narrow,
because N decides how many output tiles exist and therefore how many cores get
work. `[2560, 320]` is ten tiles, so ten cores of a hundred and thirty do all of
it. That is invariant 38, and the linears are ~43 ms of an ~83 ms step -- the
largest effect in the model that has never been attacked.

The evidence that width is the whole story: `hc_up` `[320, 10240]` does four
times the parameters of `hc_down` `[2560, 320]` in half the time, 14.45 us
against 31.50.

So give the idle cores a slice of the reduction. Core (g, nt) accumulates only
K-tiles `[kt_lo, kt_hi)` for output column nt and writes a partial into
`[1, G, M, N]`; the host sums over G with one `ttnn.sum`. No cross-core
semaphore and no second kernel pass -- the reduction rides on a stock op.

Correctness against `ttnn.linear` first, then the time.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import ttnn

KDIR = Path(__file__).resolve().parent / "kernels"
TILE = 32

# (label, K, N, weight dtype) -- the narrow shapes the model actually runs.
SHAPES = [
    ("hc_down  [2560, 320]", 2560, 320, ttnn.bfloat8_b),
    ("router   [2560, 512]", 2560, 512, ttnn.float32),
    ("qsa out  [1536, 2560]", 1536, 2560, ttnn.bfloat8_b),
]


def build(a, w, out, grid, kt: int, nt: int, groups: int, fp32: bool = True):
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    acc = {}
    for tag, t in (("a", a), ("w", w), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"{tag}: expected an interleaved tensor")
        acc[tag] = ct

    # One core per (reduction group, output column). Cores past that idle.
    plan = []
    for g in range(groups):
        lo, hi = (kt * g) // groups, (kt * (g + 1)) // groups
        for n in range(nt):
            plan.append((lo, hi, n, g))
    if len(plan) > len(cores):
        raise RuntimeError(f"{len(plan)} work items over {len(cores)} cores")
    n_active = len(plan)
    while len(plan) < len(cores):
        plan.append((0, 0, 0, 0))          # idle: reads nothing, writes nothing

    cbs = [
        ttnn.CBDescriptor(
            total_size=4 * acc["a"][1], core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=0, data_format=a.dtype, page_size=acc["a"][1])]),
        ttnn.CBDescriptor(
            total_size=4 * acc["w"][1], core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=1, data_format=w.dtype, page_size=acc["w"][1])]),
        ttnn.CBDescriptor(
            total_size=2 * acc["o"][1], core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=16, data_format=out.dtype, page_size=acc["o"][1])]),
    ]

    def kern(name, ct, args, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=str(KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfg)

    a_addr, w_addr, o_addr = a.buffer_address(), w.buffer_address(), out.buffer_address()
    return ttnn.ProgramDescriptor(
        kernels=[
            kern("ksplit_reader.cpp", [kt, nt] + acc["a"] + acc["w"],
                 [[a_addr, w_addr, lo, hi, n] for lo, hi, n, _ in plan],
                 ttnn.ReaderConfigDescriptor()),
            kern("ksplit_compute.cpp", [], [[hi - lo] for lo, hi, _, _ in plan],
                 ttnn.ComputeConfigDescriptor(
                     math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)),
            kern("ksplit_writer.cpp", [nt] + acc["o"],
                 [[o_addr, g, n, int(i < n_active)]
                  for i, (_, _, n, g) in enumerate(plan)],
                 ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[], cbs=cbs)


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dev"))
    from _device_model import open_model                              # noqa: E402

    mesh, cfg, m = open_model(max_seq_len=512)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    torch.manual_seed(0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        n_cores = grid.x * grid.y
        print(f"RESULT grid {grid.x}x{grid.y} = {n_cores} cores", flush=True)

        for label, K, N, wdt in SHAPES:
            kt, nt = K // TILE, N // TILE
            groups = max(1, min(kt, n_cores // nt))
            a = ttnn.from_torch(torch.randn(1, 1, TILE, K) * 0.05, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.02, dtype=wdt,
                                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            # bfloat16 partials, which is the configuration that shows the win.
            # float32 partials with fp32_dest_acc_en were tried: the error
            # improved only from 1.72e-02 to 1.29e-02 while hc_down went 14.83 ->
            # 31.20 us, so the accuracy is not mainly the partials' precision and
            # buying it costs the whole speedup. Both numbers are in handoff 14.
            out = ttnn.from_torch(torch.zeros(1, groups, TILE, N), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            try:
                prog = build(a, w, out, grid, kt, nt, groups)
            except Exception as exc:                                  # noqa: BLE001
                print(f"RESULT {label}: not built: {exc}", flush=True)
                continue

            def ksplit():
                ttnn.generic_op([a, w, out], prog)
                return ttnn.sum(out, dim=1, keepdim=True)

            ttnn.synchronize_device(mesh)
            got = ttnn.to_torch(ksplit(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
            ref = ttnn.to_torch(ttnn.linear(a, w),
                                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]

            # Against each other *and* against float64 on the quantised operands.
            # "Differs from ttnn.linear" is not the same as "less accurate": at
            # bfloat8_b, two roundings are already ~1.6e-02, so the question is
            # whether the kernel is further from the truth than the op it would
            # replace, not whether it matches it bit for bit.
            a64 = ttnn.to_torch(a, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
            w64 = ttnn.to_torch(w, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
            truth = (a64.to(torch.float64).squeeze(0).squeeze(0)
                     @ w64.to(torch.float64).squeeze(0).squeeze(0))
            scale = truth.abs().max().item()

            def rel_to_truth(t):
                v = t.to(torch.float64).squeeze(0).squeeze(0)
                return (v - truth).abs().max().item() / max(scale, 1e-30)

            r_mine, r_ttnn = rel_to_truth(got), rel_to_truth(ref)
            rel = (got.to(torch.float64) - ref.to(torch.float64)).abs().max().item() / max(scale, 1e-30)
            verdict = ("as accurate" if r_mine <= r_ttnn * 1.5
                       else f"{r_mine / max(r_ttnn, 1e-30):.1f}x worse")
            print(f"RESULT {label} groups={groups:3d} cores={groups*nt:4d}", flush=True)
            print(f"RESULT   vs float64: kernel {r_mine:.2e}, ttnn.linear {r_ttnn:.2e} "
                  f"-> {verdict};  they differ from each other by {rel:.2e}", flush=True)
            ok = r_mine <= r_ttnn * 1.5

            def timed(fn, tag, reps=30, iters=8):
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
                print(f"RESULT   {tag:28s} {us:8.2f} us", flush=True)
                return us

            base = timed(lambda: ttnn.linear(a, w), "ttnn.linear")
            mine = timed(ksplit, "k-split kernel + sum")
            print(f"RESULT   -> {base / mine:.2f}x", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
