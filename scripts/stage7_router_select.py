"""The routing tail as one kernel: correctness against the chain, then the time.

    uv run python scripts/stage7_router_select.py

The router chain is 9.35 ms of a 82.11 ms step and `decode_ablation_check.py`
prices its pieces: the global `ttnn.topk` over 512 experts is 5.00 ms, the
normalise 0.93, `mesh_partition` 0.73, and the local `topk` inside
`wide_expert_ffn` another 2.00 -- 8.66 ms for work on 512 numbers.

Removing four of the chain's ops moved the step 0.24 ms, so this is not op
overhead; it is those two sorts. `router_select.cpp` does the whole tail on one
core in unsigned-integer compares (after a softmax every probability is
positive, and positive IEEE floats order as their bit patterns do).

Checked against the chain it replaces *and* against a float64 model of the same
rule, because "differs from the ttnn ops" and "wrong" are not the same claim
(invariant 57).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import ttnn

KDIR = Path(__file__).resolve().parent / "kernels"
E_TOTAL, TOP_K, K_SEL = 512, 10, 10
ROWS = 32          # the tile's height; only row 0 is real at decode


def build(probs, devid, vals, idx, m_rows: int, e_local: int):
    dev = probs.device()
    core = ttnn.CoreCoord(0, 0)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    acc = {}
    for tag, t in (("p", probs), ("d", devid), ("v", vals), ("i", idx)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"{tag}: expected an interleaved tensor")
        acc[tag] = ct

    p_bf16 = 1 if probs.dtype == ttnn.bfloat16 else 0
    v_bf16 = 1 if vals.dtype == ttnn.bfloat16 else 0
    # A CB's total size must be a whole number of its page size.
    probs_cb = acc["p"][1] * (E_TOTAL // 32 + 2)
    misc_cb = 64 * ((4 * acc["v"][1] + 4 * acc["i"][1] + 512 + 63) // 64)

    cbs = [
        ttnn.CBDescriptor(
            total_size=probs_cb, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=0, data_format=probs.dtype, page_size=acc["p"][1])]),
        ttnn.CBDescriptor(
            total_size=misc_cb, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=1, data_format=ttnn.uint32, page_size=64)]),
    ]
    ct = [E_TOTAL, e_local, TOP_K, K_SEL, m_rows, p_bf16, v_bf16,
          acc["p"][1], acc["v"][1], acc["i"][1]]
    ct += acc["p"] + acc["d"] + acc["v"] + acc["i"]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "router_select.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs, compile_time_args=ct,
        runtime_args=[(core, [probs.buffer_address(), devid.buffer_address(),
                              vals.buffer_address(), idx.buffer_address()])],
        config=ttnn.ReaderConfigDescriptor())
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)


def golden(probs_host: torch.Tensor, dev: int, e_local: int):
    """The rule the chain implements, in float64: threshold, admit ties, normalise."""
    p = probs_host.to(torch.float64)
    vals_top = torch.topk(p, TOP_K, dim=-1, largest=True, sorted=True).values
    thr = vals_top[..., TOP_K - 1: TOP_K]
    keep = (p >= thr).to(torch.float64)
    kept = p * keep
    w = kept / kept.sum(dim=-1, keepdim=True)
    lo = dev * e_local
    w_local = w[..., lo: lo + e_local]
    v, i = torch.topk(w_local, K_SEL, dim=-1, largest=True, sorted=True)
    return v, i


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dev"))
    from _device_model import open_model                              # noqa: E402

    mesh, cfg, m = open_model(max_seq_len=512)
    n_dev = mesh.get_num_devices()
    e_local = E_TOTAL // n_dev
    rep = ttnn.ReplicateTensorToMesh(mesh)
    torch.manual_seed(0)
    try:
        print(f"RESULT {n_dev} devices, E_TOTAL {E_TOTAL}, E_LOCAL {e_local}, "
              f"TOP_K {TOP_K}, K_SEL {K_SEL}", flush=True)

        # Logits shaped like the router's own output, then the model's softmax.
        logits_host = torch.randn(1, 1, ROWS, E_TOTAL) * 2.0
        logits = ttnn.from_torch(logits_host, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        probs = ttnn.softmax(logits, dim=-1)
        probs_host = ttnn.to_torch(probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]

        devid = ttnn.from_torch(
            torch.arange(n_dev, dtype=torch.int32).reshape(n_dev, 1, 1, 1).expand(
                n_dev, 1, 1, 16).contiguous(),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        vals = ttnn.from_torch(torch.zeros(1, 1, ROWS, 32), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        idx = ttnn.from_torch(torch.zeros(1, 1, ROWS, 32), dtype=ttnn.uint16,
                              layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

        for m_rows in (1, ROWS):
            prog = build(probs, devid, vals, idx, m_rows, e_local)
            ttnn.generic_op([probs, devid, vals, idx], prog)
            ttnn.synchronize_device(mesh)
            gv = ttnn.to_torch(vals, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            gi = ttnn.to_torch(idx, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

            # Only the *carrying* slots can be wrong in a way that matters. The
            # top-10 is global and the experts are split four ways, so a device
            # typically holds two or three of them and the other seven K_SEL
            # slots are zero-weight padding -- there `ttnn.topk` returns
            # whichever of the 125 tied zeros it likes and this kernel returns
            # id 0, and both are multiplied by a zero weight before anything
            # reads them. Comparing padding indices measures the tie-break, not
            # the routing.
            worst_v, bad_i, checked, carrying = 0.0, 0, 0, 0
            for d in range(n_dev):
                wv, wi = golden(probs_host, d, e_local)
                for r in range(m_rows):
                    checked += 1
                    got_v = gv[d, 0, r, :K_SEL].to(torch.float64)
                    got_i = gi[d, 0, r, :K_SEL].to(torch.int64)
                    ref_v, ref_i = wv[0, 0, r], wi[0, 0, r]
                    worst_v = max(worst_v, (got_v - ref_v).abs().max().item())
                    live = ref_v > 0
                    carrying += int(live.sum().item())
                    bad_i += int(((got_i != ref_i) & live).sum().item())
            print(f"RESULT M={m_rows:2d}: {checked} rows checked; vs float64 max weight "
                  f"error {worst_v:.3e}; {bad_i} wrong indices out of {carrying} "
                  f"weight-carrying slots (of {checked * K_SEL} total)", flush=True)

        # --- against the ttnn chain it replaces, and the clock ---------------
        prog = build(probs, devid, vals, idx, 1, e_local)

        def chain():
            v, _ = ttnn.topk(probs, k=TOP_K, dim=-1, largest=True, sorted=True)
            s = list(v.shape)
            thr = ttnn.slice(v, (0, 0, 0, TOP_K - 1), (s[0], s[1], s[2], TOP_K))
            keep = ttnn.ge(probs, thr, dtype=ttnn.bfloat16)
            kept = ttnn.multiply(probs, keep)
            w = ttnn.divide(kept, ttnn.sum(kept, dim=-1, keepdim=True))
            wl = ttnn.mesh_partition(w, dim=-1)
            return ttnn.topk(wl, k=K_SEL, dim=-1, largest=True, sorted=True)

        cv, ci = chain()
        ttnn.synchronize_device(mesh)
        cvh = ttnn.to_torch(cv, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        cih = ttnn.to_torch(ci, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        ttnn.generic_op([probs, devid, vals, idx], prog)
        ttnn.synchronize_device(mesh)
        gv = ttnn.to_torch(vals, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        gi = ttnn.to_torch(idx, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        gvh = gv[:, 0, 0, :K_SEL].to(torch.float64)
        cvhr = cvh[:, 0, 0, :K_SEL].to(torch.float64)
        dv = (gvh - cvhr).abs().max().item()
        live = cvhr > 0
        di = int(((gi[:, 0, 0, :K_SEL].to(torch.int64)
                   != cih[:, 0, 0, :K_SEL].to(torch.int64)) & live).sum().item())
        print(f"RESULT vs the ttnn chain at M=1: max weight diff {dv:.3e}, "
              f"{di} wrong indices out of {int(live.sum().item())} carrying slots",
              flush=True)
        # The product the MoE actually forms: sum over slots of weight * one-hot
        # expert. Equal here means the gather+matmul downstream cannot tell the
        # two paths apart, whatever the padding indices say.
        def as_vector(v, i):
            out = torch.zeros(n_dev, e_local, dtype=torch.float64)
            for d in range(n_dev):
                for sl in range(K_SEL):
                    out[d, int(i[d, 0, 0, sl])] += float(v[d, 0, 0, sl])
            return out

        mine = as_vector(gv.to(torch.float64), gi.to(torch.int64))
        theirs = as_vector(cvh.to(torch.float64), cih.to(torch.int64))
        print(f"RESULT effective per-expert weight vector: max abs diff "
              f"{(mine - theirs).abs().max().item():.3e}", flush=True)

        # Which of the two is actually closer to the rule, in float64? "Differs
        # from the ttnn ops" is not "less accurate" (invariant 57): the chain
        # divides in bfloat16 where the kernel divides in float32, so the kernel
        # can only be nearer the truth on the weights it carries.
        ref = torch.zeros(n_dev, e_local, dtype=torch.float64)
        for d in range(n_dev):
            wv, wi = golden(probs_host, d, e_local)
            for sl in range(K_SEL):
                ref[d, int(wi[0, 0, 0, sl])] += float(wv[0, 0, 0, sl])
        em = (mine - ref).abs().max().item()
        et = (theirs - ref).abs().max().item()
        verdict = ("kernel is closer" if em < et
                   else "chain is closer" if et < em else "identical")
        print(f"RESULT vs float64 weight vector: kernel {em:.3e}, chain {et:.3e} "
              f"-> {verdict}", flush=True)

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
            print(f"RESULT {tag:34s} {us:8.2f} us  ({48 * us / 1000:6.2f} ms a token)",
                  flush=True)
            return us

        a = timed(chain, "topk + normalise + partition + topk")
        b = timed(lambda: ttnn.generic_op([probs, devid, vals, idx], prog),
                  "router_select.cpp")
        print(f"RESULT -> {a / b:.2f}x, {48 * (a - b) / 1000:+.2f} ms a token",
              flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
