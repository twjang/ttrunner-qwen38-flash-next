"""Stage 2: gather the selected experts, then a dense matmul over K_SEL.

    uv run python scripts/stage2_gather_matmul.py --cache-dir ~/models/qwen38-tt-cache

`expert_ffn` costs **30.83 ms** a token and the SwiGLU chain behind it another
6.83, on tensors that are 97 % zeros -- 128 local experts materialised so that
~3 of them can be used. Handoff 5.2 found `sparse_matmul` zero-filling its whole
`[1, E, M, N]` output unconditionally, 1.51 GB a token before a weight is read,
and 5.4 found the expert-axis elementwise ops costing with E and not with M.

The fix does not need a matmul kernel. Stage 1 already reads exactly the
selected experts' tiles at 246-389 GB/s; make it *write* them into a compact
`[1, K_SEL, K, N]` tensor and the arithmetic becomes an ordinary dense batched
`ttnn.matmul` over an expert axis of K_SEL. That removes the zero-fill, shrinks
every downstream elementwise op by E/K_SEL, and drops `sparse_matmul` entirely.

This measures the two paths against each other on the real layer-0 weights, and
checks the gather against a torch golden first -- a fast wrong answer is not a
result.

K_SEL is fixed at the worst case (10: every one of the top-10 could land on one
device) because a trace needs a static shape. The expected count per device is
2.5, so the padding is real waste -- and still far less than 128.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ttrunner_qwen38_flash_next.tt.weights import TTWeights          # noqa: E402

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "expert_gather.cpp")
IDX_LEN = 128          # index page width in uint32 -> 512 B
K_SEL = 3              # the real per-device budget, ceil(10/4)
READ_BATCH = 8


def core_list(grid):
    return [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]


def split_work(total: int, n: int):
    return [((total * c) // n, (total * (c + 1)) // n) for c in range(n)]


def build_gather(weights, indices, out, grid, k_sel: int, wide: bool = True):
    cores = core_list(grid)
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )
    accessors = {}
    for tag, t in (("w", weights), ("i", indices), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"{tag}: expected an interleaved tensor, got {len(ct)} CT args")
        accessors[tag] = ct

    tile_bytes, idx_bytes = accessors["w"][1], accessors["i"][1]
    if accessors["o"][1] != tile_bytes:
        raise RuntimeError(
            f"output page {accessors['o'][1]} != weight page {tile_bytes}; the gather "
            "copies whole tiles, so both tensors must have the same dtype"
        )

    kt, nt = weights.shape[-2] // 32, weights.shape[-1] // 32
    tiles_per_expert = kt * nt
    total_work = k_sel * tiles_per_expert

    cb_w = ttnn.CBDescriptor(
        total_size=(READ_BATCH + 1) * tile_bytes, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=0, data_format=weights.dtype, page_size=tile_bytes)],
    )
    cb_aux = ttnn.CBDescriptor(
        total_size=64 * ((idx_bytes + 64 + 63) // 64), core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=1, data_format=ttnn.uint32, page_size=64)],
    )

    # 0..4, then K_SEL, NT and WIDE, so the kernel's TensorAccessorArgs<8> lands right.
    # K_SEL also separates program-cache entries: generic_op hashes compile-time
    # args by value but runtime args only by count (handoff 4g).
    ct_args = [tiles_per_expert, tile_bytes, READ_BATCH, idx_bytes,
               weights.shape[1], k_sel, nt, 1 if wide else 0]
    ct_args += accessors["w"] + accessors["i"] + accessors["o"]

    addrs = (weights.buffer_address(), indices.buffer_address(), out.buffer_address())
    runtime_args = [
        (core, [*addrs, lo, hi])
        for core, (lo, hi) in zip(cores, split_work(total_work, len(cores)))
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs, compile_time_args=ct_args, runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb_w, cb_aux])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--name", default="ffn_gateup_exps.weight")
    args = ap.parse_args()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 << 20)
    replicate = ttnn.ReplicateTensorToMesh(mesh)
    try:
        grid = mesh.compute_with_storage_grid_size()
        store = TTWeights(args.cache_dir, mesh)
        w = store.blk(args.layer, args.name)
        E, K, N = w.shape[1], w.shape[2], w.shape[3]
        print(f"RESULT weights {tuple(w.shape)} {w.dtype}, grid {grid.x}x{grid.y}", flush=True)

        ids = [(i * 37) % E for i in range(K_SEL)]
        assert len(set(ids)) == K_SEL and K_SEL <= IDX_LEN
        host_idx = torch.zeros(1, 1, 1, IDX_LEN, dtype=torch.int32)
        for i, e in enumerate(ids):
            host_idx[0, 0, 0, i] = e
        idx_dev = ttnn.from_torch(host_idx, dtype=ttnn.uint32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                  mesh_mapper=replicate)
        # Wide, not compact: the experts sit side by side on the output axis so
        # one `ttnn.linear` covers all of them (see the kernel header).
        gathered = ttnn.from_torch(torch.zeros(1, 1, K, K_SEL * N), dtype=w.dtype,
                                   layout=ttnn.TILE_LAYOUT, device=mesh,
                                   mesh_mapper=replicate)

        prog = build_gather(w, idx_dev, gathered, grid, K_SEL)

        # --- correctness first ---------------------------------------------
        ttnn.generic_op([w, idx_dev, gathered], prog)
        ttnn.synchronize_device(mesh)
        got = ttnn.to_torch(gathered, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
        ref_full = ttnn.to_torch(w, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
        want = torch.cat([ref_full[:, e, :, :] for e in ids], dim=-1).unsqueeze(1)
        err = (got.to(torch.float64) - want.to(torch.float64)).abs().max().item()
        print(f"RESULT gather vs torch index: max abs err {err:.3e} "
              f"{'EXACT' if err == 0.0 else 'MISMATCH'}", flush=True)
        if err != 0.0:
            raise SystemExit("gather is wrong; not timing a wrong answer")

        # --- the two paths ---------------------------------------------------
        x1 = ttnn.from_torch(torch.randn(1, 1, 1, K) * 0.05, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=replicate)
        xe = ttnn.repeat(x1, (1, E, 1, 1))
        spars = ttnn.from_torch(torch.zeros(1, 1, 1, E), dtype=ttnn.bfloat16,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                mesh_mapper=replicate)

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
            ms = best / reps * 1e3
            print(f"RESULT {label:44s} {ms:8.4f} ms  -> {48 * ms:7.2f} ms/48 layers",
                  flush=True)
            return ms

        g = timed(lambda: ttnn.generic_op([w, idx_dev, gathered], prog), "gather only")
        mm = timed(lambda: ttnn.linear(x1, gathered),
                   f"one wide matmul [{K}, {K_SEL}x{N}]")

        def both():
            ttnn.generic_op([w, idx_dev, gathered], prog)
            return ttnn.linear(x1, gathered)

        tot = timed(both, "gather + dense matmul")

        try:
            sp = timed(
                lambda: ttnn.sparse_matmul(
                    xe, w, sparsity=spars, nnz=K_SEL,
                    is_input_a_sparse=True, is_input_b_sparse=True),
                f"sparse_matmul over E={E} (today)")
            print(f"RESULT ---", flush=True)
            print(f"RESULT {sp:.4f} -> {tot:.4f} ms a call ({sp / tot:.2f}x), "
                  f"{48 * (sp - tot):+.2f} ms a token over 48 layers", flush=True)
        except Exception as exc:                                    # noqa: BLE001
            print(f"RESULT sparse_matmul comparison rejected: {type(exc).__name__}: "
                  f"{(str(exc) or repr(exc)).splitlines()[0][:160]}", flush=True)
            print(f"RESULT gather+matmul is {tot:.4f} ms a call, "
                  f"{48 * tot:.2f} ms over 48 layers, against 30.83 measured for "
                  f"expert_ffn today", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
