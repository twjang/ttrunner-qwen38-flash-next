#!/usr/bin/env python
"""Stage 1: measure the DRAM read rate of an index-driven expert gather.

This is a MEASUREMENT, not a working MoE. It runs the smallest ttnn.generic_op
program that (a) takes the per-device expert weight tensor as an io_tensor,
(b) takes the selected local expert ids as an io_tensor -- not as host runtime
args, so a captured trace stays valid when the routing changes, (c) reads only
the tile pages belonging to those experts, and (d) folds them into two uint32
accumulators per core written into a small pre-allocated output tensor.

Two numbers decide whether stage 2 is worth building:

  * ms/layer at the real per-device expert budget (ceil(10/4) = 3 local experts
    for gateup + down), against the 0.033 ms/layer floor and the 0.64 ms/layer
    the current sparse_matmul path costs.
  * the sparse/dense time ratio against the same kernel with all 128 experts
    selected. That control run is measured on this hardware with this kernel,
    so it is the honest ceiling: if sparse time / dense time is close to
    K_SEL/128, the read really is proportional to the bytes asked for.

Run:  uv run python scripts/stage1_expert_gather_bw.py --cache-dir ~/models/qwen38-tt-cache
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import ttnn

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "expert_gather_checksum.cpp")

IDX_LEN = 128  # index tensor width in uint32 -> 512 B page. K_SEL <= IDX_LEN.
OUT_ROW = 32  # uint32 per core -> 128 B page, a multiple of DRAM_ALIGNMENT(64)

TENSORS = {
    # name in the weight cache            -> (K, N, ttnn dtype)
    "ffn_gateup_exps.weight": (2560, 1280, "bfloat4_b"),
    "ffn_down_exps.weight": (640, 2560, "bfloat8_b"),
}


def core_list(grid) -> list:
    return [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]


def split_work(total: int, n: int) -> list[tuple[int, int]]:
    """Contiguous, balanced work ranges. Sums to exactly `total`."""
    return [((total * c) // n, (total * (c + 1)) // n) for c in range(n)]


def build_program(weights, indices, out, grid, k_sel: int, read_batch: int):
    """One reader kernel over the whole worker rectangle. No compute kernel."""
    cores = core_list(grid)
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )

    # For an interleaved tensor TensorAccessorArgs emits exactly two compile-time
    # args: the config word and the ALIGNED page size. We reuse [1] as the page
    # stride, which is what noc_async_read_page actually transfers.
    accessors = {}
    for tag, t in (("w", weights), ("i", indices), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(
                f"{tag}: expected an interleaved tensor (2 accessor CT args), got {len(ct)}. "
                "A sharded tensor emits more and the kernel's CTA offsets would shift."
            )
        accessors[tag] = ct

    tile_bytes = accessors["w"][1]
    idx_bytes = accessors["i"][1]
    out_bytes = accessors["o"][1]

    kt, nt = weights.shape[-2] // 32, weights.shape[-1] // 32
    tiles_per_expert = kt * nt
    total_work = k_sel * tiles_per_expert

    # cb_0: the weight landing buffer. One page of slop pays for the in-kernel
    # 64 B alignment bump (a CB base is only guaranteed 16 B aligned, but a
    # Blackhole DRAM read needs local & 63 == noc & 63).
    cb_w = ttnn.CBDescriptor(
        total_size=(read_batch + 1) * tile_bytes,
        core_ranges=crs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0, data_format=weights.dtype, page_size=tile_bytes
            )
        ],
    )
    # cb_1: index page + output page + 64 B of alignment slop, as 64 B pages.
    aux_total = 64 * ((idx_bytes + out_bytes + 64 + 63) // 64)
    cb_aux = ttnn.CBDescriptor(
        total_size=aux_total,
        core_ranges=crs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.uint32, page_size=64)
        ],
    )

    n_experts = weights.shape[1]
    ct_args = [tiles_per_expert, tile_bytes, read_batch, idx_bytes, out_bytes, n_experts]
    ct_args += accessors["w"] + accessors["i"] + accessors["o"]
    # `generic_op`'s program hash covers compile_time_args by value but
    # runtime_args only by count, so k_sel has to be here or every K in the
    # sweep reuses the first compiled program and re-runs its work ranges.
    # Trailing and unread by the kernel, which keeps the accessor offsets put.
    ct_args.append(k_sel)

    w_addr, i_addr, o_addr = (
        weights.buffer_address(),
        indices.buffer_address(),
        out.buffer_address(),
    )
    runtime_args = [
        (core, [w_addr, i_addr, o_addr, lo, hi, page])
        for page, (core, (lo, hi)) in enumerate(zip(cores, split_work(total_work, len(cores))))
    ]

    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=ct_args,
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb_w, cb_aux])
    return prog, tiles_per_expert, tile_bytes, len(cores)


def write_indices(mesh, replicate, idx_dev, ids: list[int]) -> None:
    """Push a new selection list into the SAME device buffer.

    This is the whole point of passing the indices as an io_tensor: under trace
    replay the buffer address is frozen but its contents are not, so routing
    changes without a re-capture.
    """
    host = torch.zeros(1, 1, 1, IDX_LEN, dtype=torch.int32)
    host[0, 0, 0, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate
        ),
        idx_dev,
    )


def verify(mesh, out, n_cores: int, ids: list[int], tiles_per_expert: int) -> tuple[bool, str]:
    """`sum_tid` is exactly predictable, so this proves the whole address path.

    Every core folds `tile_id` for each page it reads, so the total over all
    cores is a closed form. If it matches, every intended tile was visited
    exactly once and nothing else was.

    `sum_data` carries the other half: the kernel poisons each landing slot
    before reading it, so this checks that the fold is neither zero nor the
    sentinel, and that the four devices disagree -- each holds a different 128
    experts, so agreement would mean they all read the same bytes.

    `k_sel == 0` is legal and measures the per-launch floor: no tiles, so the
    data checks are skipped and only the "read nothing" bookkeeping is asserted.
    """
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    got = got.reshape(-1, n_cores, OUT_ROW).to(torch.int64)  # [n_dev, cores, 32]

    tpe = tiles_per_expert
    want_tid = (tpe * tpe * sum(ids) + len(ids) * (tpe * (tpe - 1) // 2)) % (1 << 32)
    want_n = len(ids) * tpe

    ok = True
    parts = []
    datas = []
    for d in range(got.shape[0]):
        sum_data = int(got[d, :, 0].sum() % (1 << 32))
        sum_tid = int(got[d, :, 1].sum() % (1 << 32))
        n_tiles = int(got[d, :, 2].sum())
        datas.append(sum_data)
        ok = ok and sum_tid == want_tid and n_tiles == want_n
        if want_n:
            ok = ok and sum_data != 0
        parts.append(f"d{d}:tiles={n_tiles},tid=0x{sum_tid:08x},data=0x{sum_data:08x}")

    # The docstring's other promise, which was never actually checked: each
    # device holds a different 128 experts, so identical sums across devices
    # would mean every device read the same bytes -- the one failure mode the
    # data checksum exists to catch.
    if want_n and len(datas) > 1 and len(set(datas)) != len(datas):
        ok = False
        parts.append("FAIL:devices agree on sum_data (all read the same bytes?)")

    # The kernel poisons each landing slot with 0xDEADBExx before reading it, so
    # a fold that comes back looking like the sentinel means the read never
    # landed and `sum_data != 0` would have passed on garbage.
    if want_n and any((v & 0xFFFFFF00) == 0xDEADBE00 for v in datas):
        ok = False
        parts.append("FAIL:sum_data looks like the poison sentinel -- reads did not land")

    return ok, f"want tiles={want_n} tid=0x{want_tid:08x} | " + " ".join(parts)


def time_traced(mesh, io, prog, reps: int, iters: int) -> list[float]:
    ttnn.generic_op(io, prog)  # warm the program cache -- capture fails otherwise
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(reps):
        ttnn.generic_op(io, prog)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) / reps)
    ttnn.release_trace(mesh, tid)
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 10, 128])
    ap.add_argument("--read-batch", type=int, default=8)
    ap.add_argument("--reps", type=int, default=50, help="generic_op calls per trace")
    ap.add_argument("--iters", type=int, default=20, help="trace replays timed")
    ap.add_argument("--grid", type=int, nargs=2, default=None, help="override worker grid x y")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ttrunner_qwen38_flash_next.tt.weights import TTWeights

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 << 20)
    replicate = ttnn.ReplicateTensorToMesh(mesh)
    try:
        grid = (
            ttnn.CoreCoord(*args.grid) if args.grid else mesh.compute_with_storage_grid_size()
        )
        n_cores = grid.x * grid.y
        print(f"worker grid {grid.x}x{grid.y} = {n_cores} cores, read_batch={args.read_batch}\n")

        store = TTWeights(args.cache_dir, mesh)

        idx_dev = ttnn.from_torch(
            torch.zeros(1, 1, 1, IDX_LEN, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=replicate,
        )
        out_dev = ttnn.from_torch(
            torch.zeros(1, 1, n_cores, OUT_ROW, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=replicate,
        )

        per_k_layer_ms: dict[int, float] = {}
        for k_sel in args.k:
            ids = [(i * 37) % 128 for i in range(k_sel)]
            # 37 is coprime with 128, so this is a permutation only while
            # k_sel <= 128; past that ids repeat (double-counting the byte
            # figure) and the write would run off the 512 B index page.
            assert k_sel <= IDX_LEN, f"k_sel {k_sel} exceeds the index page ({IDX_LEN})"
            assert len(set(ids)) == k_sel, f"repeated expert ids at k_sel={k_sel}"
            write_indices(mesh, replicate, idx_dev, ids)
            ttnn.synchronize_device(mesh)

            layer_ms = 0.0
            for suffix, (K, N, dtype) in TENSORS.items():
                w = store.blk(args.layer, suffix)
                assert tuple(w.shape) == (1, 128, K, N), f"{suffix}: got {tuple(w.shape)}"

                prog, tpe, tile_bytes, ncore = build_program(
                    w, idx_dev, out_dev, grid, k_sel, args.read_batch
                )
                io = [w, idx_dev, out_dev]

                ttnn.generic_op(io, prog)
                ttnn.synchronize_device(mesh)
                ok, msg = verify(mesh, out_dev, ncore, ids, tpe)
                flag = "ok " if ok else "BAD"
                print(f"  [{flag}] K={k_sel:3d} {suffix:24s} {msg}")
                if not ok:
                    raise SystemExit("address path is wrong -- do not trust the timings")

                s = time_traced(mesh, io, prog, args.reps, args.iters)
                ms = statistics.median(s) * 1e3
                best = min(s) * 1e3
                nbytes = k_sel * tpe * tile_bytes
                gbs = nbytes / (ms * 1e-3) / 1e9
                layer_ms += ms
                print(
                    f"        {nbytes/1e6:8.2f} MB/device  median {ms:7.4f} ms  "
                    f"min {best:7.4f} ms  {gbs:7.1f} GB/s/device"
                )
            per_k_layer_ms[k_sel] = layer_ms
            print(f"        -> gateup+down = {layer_ms:.4f} ms/layer, "
                  f"{48 * layer_ms:.2f} ms for 48 layers\n")

        print("summary (per device, gateup+down):")
        dense = per_k_layer_ms.get(128)
        for k_sel, ms in sorted(per_k_layer_ms.items()):
            line = f"  K={k_sel:3d}  {ms:8.4f} ms/layer  {48*ms:8.2f} ms/48 layers"
            if dense and k_sel != 128:
                line += f"  ratio_vs_dense={ms/dense:.4f} (ideal {k_sel/128:.4f})"
            print(line)
        print("\nreference: 0.033 ms/layer floor, 0.64 ms/layer current sparse_matmul, "
              "1.20 ms/layer ttnn moe_compute")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
