"""Gated DeltaNet on device.

Two code paths, because the two regimes want different algorithms:

**Decode (S == 1).** The delta rule is a plain recurrence, so no chunking, no
triangular system, and no host preparation are needed at all:

    state <- state * exp(g)
    delta  = (v - k @ state) * beta
    state <- state + kᵀ delta
    out    = q @ state

That is three batched matmuls over the (batch x heads) axis plus a handful of
elementwise ops, entirely on device. Using `gated_delta_attn_seq` here would mean
building `L_unit`/`L_inv` on the host every layer and round-tripping over PCIe --
far more expensive than the recurrence it replaces.

**Prefill (S > 1).** Here the chunked form wins, and `gated_delta_attn_seq` does
the expensive half. Its eight inputs are prepared on the host (see
`deltanet.prepare`), which is affordable because the cost amortises over a whole
chunk of tokens rather than one.
"""

from __future__ import annotations

import os

import ttnn

from . import ops

from .ops import HIFI4


_NO_SPLIT_KQ = bool(os.environ.get("TT_NO_SPLIT_KQ"))
_NO_BCAST_OUTER = bool(os.environ.get("TT_NO_BCAST_OUTER"))
# The delta rule's tail as one launch a head.
_NO_FUSED_DELTA_OUT = bool(os.environ.get("TT_NO_FUSED_DELTA_OUT"))
# One megabyte of state, in tiles of a float32 32x32. Above this, reading it
# twice costs more than the concat that avoids it.
_SPLIT_KQ_MAX_TILES = 256


def _tiles(t) -> int:
    n = 1
    sh = list(t.shape)
    for d in sh[:-2]:
        n *= d
    return n * max(1, (sh[-2] + 31) // 32) * max(1, (sh[-1] + 31) // 32)


_RECUR_OUT: dict = {}
_VERIFY_WORST = {"out": 0.0, "state": 0.0, "n": 0}


def _verify(state, q, k, v, g_exp, beta, got):
    """TT_RECUR_VERIFY=1: rerun the op chain on the *same* inputs and compare.

    Handoff 45.27: the fused recurrence is right at seq 2049 and wrong at 2047
    with everything else identical, and five hypotheses for that are dead. This
    settles the remaining question -- whether the kernel produces wrong numbers
    in the model, or produces right ones that something else then corrupts.

    The op chain runs on a *copy* of the pre-call state, so it does not disturb
    the real one. Called after the fused path has already written `state`, so
    the copy has to be taken before -- see the caller.
    """
    import torch
    ref_state = _VERIFY_PRE.pop("state", None)
    if ref_state is None:
        return
    ref_out = decode_step(q, k, v, g_exp, beta, ref_state, _no_fuse=True)
    if os.environ.get("TT_RECUR_VERIFY_CONTROL") == "1":
        # Control (handoff 45.33): compare the op chain against *itself* on a
        # second clone of the same pre-call state. Both arms are then the same
        # deterministic computation, so every column must read ~0. Anything else
        # is the probe, not the kernel.
        ctrl = _VERIFY_PRE.pop("ctrl", None)
        if ctrl is not None:
            got = decode_step(q, k, v, g_exp, beta, ctrl, _no_fuse=True)
            state = ctrl
    a = ttnn.to_torch(got, mesh_composer=ttnn.ConcatMeshToTensor(got.device(), dim=0))
    b = ttnn.to_torch(ref_out, mesh_composer=ttnn.ConcatMeshToTensor(ref_out.device(), dim=0))
    n = min(a.shape[0], b.shape[0])
    do = float((a[:n].float() - b[:n].float()).abs().max())
    sa = ttnn.to_torch(state, mesh_composer=ttnn.ConcatMeshToTensor(state.device(), dim=0))
    sb = ttnn.to_torch(ref_state, mesh_composer=ttnn.ConcatMeshToTensor(ref_state.device(), dim=0))
    m = min(sa.shape[0], sb.shape[0])
    ds = float((sa[:m].float() - sb[:m].float()).abs().max())
    # Decompose the state error by term (handoff 45.35, invariant 152). The
    # kernel writes `decayed + kt (x) delta`; comparing the fused state against
    # each term separately says which one is wrong, where comparing against the
    # op chain only says that something is.
    pre = _VERIFY_PRE.pop("pristine", None)
    if pre is not None:
        def th(t):
            return ttnn.to_torch(
                t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=0)
            )[:m].to(torch.float64)
        P, G, K, V, B = th(pre), th(g_exp), th(k), th(v), th(beta)
        decayed_h = P * G
        delta_h = (V - K @ decayed_h) * B
        state_h = decayed_h + K.transpose(-2, -1) @ delta_h
        f = sa[:m].to(torch.float64)
        w2 = _VERIFY_WORST
        w2["vs_full"] = max(w2.get("vs_full", 0.0),
                            float((f - state_h).abs().max()))
        w2["vs_decayed"] = max(w2.get("vs_decayed", 0.0),
                               float((f - decayed_h).abs().max()))
        w2["outer_mag"] = max(w2.get("outer_mag", 0.0),
                              float((state_h - decayed_h).abs().max()))
        # Magnitudes, not just differences: if |fused| is ~0 while |host| is ~62,
        # the kernel is writing an empty state rather than a wrong one, and that
        # is a different bug entirely (handoff 45.36).
        w2["mag_fused"] = max(w2.get("mag_fused", 0.0), float(f.abs().max()))
        w2["mag_host"] = max(w2.get("mag_host", 0.0), float(state_h.abs().max()))
    w = _VERIFY_WORST
    w["out"] = max(w["out"], do)
    w["state"] = max(w["state"], ds)
    w["n"] += 1
    if w["n"] % 360 == 0:
        print(f"RESULT verify after {w['n']} calls: worst out {w['out']:.3e}  "
              f"worst state {w['state']:.3e}", flush=True)
        if "vs_full" in w:
            print(f"RESULT   fused state vs host full   {w['vs_full']:.3e}\n"
                  f"RESULT   fused state vs host decayed {w['vs_decayed']:.3e}\n"
                  f"RESULT   host |outer| magnitude      {w['outer_mag']:.3e}\n"
                  f"RESULT   |fused state| {w['mag_fused']:.3e}   "
                  f"|host state| {w['mag_host']:.3e}",
                  flush=True)


_VERIFY_PRE: dict = {}
_DUMPED = False


def _dump_pads(fq, fk, fv, state):
    """One-shot: what is actually in the pad regions the kernel contracts over?

    Handoff 45.34. `kt` is `[BH, 1, Dk, 1]`, so a tile's real data is column 0
    and columns 1..31 are padding; the state's outer product contracts all 32.
    `to_torch` de-tilizes and drops the pad, so this reads the *padded* view by
    asking for the tile's full width.
    """
    global _DUMPED
    _DUMPED = True
    kt = ttnn.transpose(fk, -2, -1)
    for name, t in (("fk", fk), ("fv", fv), ("kt", kt)):
        try:
            a = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=0))
            print(f"RESULT dump {name}: shape {list(t.shape)} -> torch {list(a.shape)}  "
                  f"absmax {float(a.abs().max()):.4e}  "
                  f"finite {bool(a.isfinite().all())}", flush=True)
        except Exception as exc:                                    # noqa: BLE001
            print(f"RESULT dump {name}: {type(exc).__name__}: {exc}", flush=True)
    a = ttnn.to_torch(state, mesh_composer=ttnn.ConcatMeshToTensor(state.device(), dim=0))
    print(f"RESULT dump state: absmax {float(a.abs().max()):.4e}  "
          f"finite {bool(a.isfinite().all())}", flush=True)


def _recur_out(state, v):
    """The fused recurrence writes its output into a buffer of its own.

    Kept per (state, shape) like every other `generic_op` output here: a trace
    replays a recorded graph, so a buffer it touches must not move.
    """
    # TEMPORARY probe (handoff 45.27): TT_RECUR_FRESH=1 allocates a new output
    # every call instead of reusing the cached one. Unsafe in a trace and slow,
    # but `device_quality.py` is eager -- and if the seq-2047 corruption is two
    # of these buffers aliasing, it disappears here.
    import os as _os
    if os.environ.get("TT_RECUR_FRESH") == "1":
        import torch as _t
        return ttnn.from_torch(
            _t.zeros(*[int(d) for d in v.shape]), dtype=v.dtype,
            layout=ttnn.TILE_LAYOUT, device=v.device(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(v.device()))
    key = (id(state), tuple(v.shape), str(v.dtype))
    got = _RECUR_OUT.get(key)
    if got is None:
        import torch
        got = ttnn.from_torch(
            torch.zeros(*[int(d) for d in v.shape]), dtype=v.dtype,
            layout=ttnn.TILE_LAYOUT, device=v.device(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(v.device()))
        _RECUR_OUT[key] = got
    return got


def decode_step(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    g_exp: ttnn.Tensor,
    beta: ttnn.Tensor,
    state: ttnn.Tensor,
    _no_fuse: bool = False,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """One recurrent step.

    q, k:    [BH, 1, 1, Dk]  (already l2-normalised and q scaled by Dk**-0.5)
    v:       [BH, 1, 1, Dv]
    g_exp:   [BH, 1, 1, 1]   exp of the log decay
    beta:    [BH, 1, 1, 1]
    state:   [BH, 1, Dk, Dv]  -- updated **in place**

    `state` is written in place rather than replaced. That keeps the recurrent
    state at a fixed device address across steps, which is what allows the whole
    48-layer step to be captured as one ttnn trace (a trace replays a recorded
    graph, so any buffer it touches must not move), and it avoids reallocating
    3.1 MB per layer per token along the way.

    Returns the output; the new state is in `state`.
    """
    # The whole recurrence in one launch when the shapes and dtypes allow: it
    # reads the state once and writes it once, where everything below makes five
    # passes over it. 2.95 ms a token (handoff 45.23), and `recur_check.py` puts
    # it at 1.09e-03 from float64 against the chain's 8.26e-04.
    # The dtype check comes **first**, before the transpose and before
    # `_recur_out` allocates: Python evaluates arguments eagerly, so calling
    # `fused_recurrence(..., ttnn.transpose(k), _recur_out(...))` pays both even
    # when the guard inside declines -- and `_recur_out`'s first allocation then
    # lands inside a trace capture, which is the "allocating device buffers is
    # unsafe" hazard. In the model the state is float32 and q/k/v arrive
    # bfloat16, so the decline is the common path, not the rare one.
    if _no_fuse:
        pass
    elif not ops._NO_FUSED_RECUR:
        if os.environ.get("TT_RECUR_VERIFY") == "1":
            _VERIFY_PRE["state"] = ttnn.clone(state)
            _VERIFY_PRE["pristine"] = ttnn.clone(state)
            if os.environ.get("TT_RECUR_VERIFY_CONTROL") == "1":
                _VERIFY_PRE["ctrl"] = ttnn.clone(state)
        # The model hands this bfloat16 q/k/v/g/beta against a float32 state, and
        # one compute kernel configures its unpacker from one circular buffer
        # (invariant 76), so they have to agree. Casting here is the *measurable*
        # form, not the shipping one: five casts on 48-tile tensors, ~0.9 ms a
        # token, against the 2.95 ms the fusion targets. The shipping form is
        # `fused_qkv_heads` and `fused_delta_scalars` emitting the state's dtype
        # in the first place, which costs nothing -- do that once this is timed.
        # `ttnn.typecast` allocates at capture and the replay reuses it, so it is
        # trace-safe in a way a `generic_op` output is not.
        fq, fk, fv, fg, fb = (
            t if t.dtype == state.dtype else ttnn.typecast(t, state.dtype)
            for t in (q, k, v, g_exp, beta))
        if os.environ.get("TT_RECUR_DUMP") == "1" and not _DUMPED:
            _dump_pads(fq, fk, fv, state)
        fused_all = ops.fused_recurrence(
            state, fq, fk, ttnn.transpose(fk, -2, -1), fv, fg, fb,
            # `fv`, so the output CB is the state's dtype. Allocating it from
            # the *uncast* `v` makes cb_out bfloat16 and the compute then packs
            # a float32 dst into it -- which hung, in the same family as
            # `ksgemv_compute`'s note that packing into a Float32 CB hangs here.
            # The cost is that `reinject` downstream sees a float32 branch and
            # declines ("wants one dtype for hyper, branch and out"), so this
            # form is measurable but not shippable: the shipping fix is upstream
            # dtypes, not a narrowing pack.
            _recur_out(state, fv))
        if fused_all is not None:
            if os.environ.get("TT_RECUR_VERIFY") == "1":
                _verify(state, q, k, v, g_exp, beta, fused_all)
            return fused_all

    decayed = ttnn.multiply(state, g_exp)

    # `k @ decayed` and `q @ decayed` are two reads of the same ~100 MB state at
    # B=32, and the state matmuls are memory-bound (~35 GB/s), so they are fused
    # into one by stacking [k; q]. That is exact, not an approximation:
    #
    #   out = q @ (decayed + kᵀ delta) = q@decayed + (q @ kᵀ) delta
    #                                  = q@decayed + (q·k) delta
    #
    # since q and k are single rows, so q @ kᵀ is a scalar per head.
    #
    # It stops paying once the state is small. `chain_price.py`: the concat is
    # **20.15 us** -- a two-row stack of [BH,1,1,Dk] tensors is a sub-tile row
    # interleave, which is data movement, not arithmetic -- and the matmul costs
    # 23.59 us at two rows against 23.86 at one, so the second matmul is nearly
    # free. With the two slices that the split also removes (5.26 us each), one
    # matmul plus a concat plus two slices is 54.2 us where two matmuls are 47.7.
    # The saving is the concat; the crossover is where reading the state twice
    # costs more than that, which at 388 GB/s is around a megabyte.
    small = _tiles(state) <= _SPLIT_KQ_MAX_TILES
    if _NO_SPLIT_KQ or not small:
        kq = ttnn.concat([k, q], dim=-2)                              # [BH,1,2,Dk]
        both = ttnn.matmul(kq, decayed, compute_kernel_config=HIFI4)  # [BH,1,2,Dv]
        shape = list(both.shape)
        v_dim = shape[-1]
        predicted = ttnn.slice(both, (0, 0, 0, 0), (shape[0], shape[1], 1, v_dim))
        q_decayed = ttnn.slice(both, (0, 0, 1, 0), (shape[0], shape[1], 2, v_dim))
    else:
        predicted = ttnn.matmul(k, decayed, compute_kernel_config=HIFI4)
        q_decayed = ttnn.matmul(q, decayed, compute_kernel_config=HIFI4)

    # Six ops -- a subtract, two multiplies, a reduction and two more -- in one
    # launch when the shapes allow. `fused_delta_out` returns both `delta`, which
    # the state update below still needs, and the step's output.
    fused = None
    if not _NO_FUSED_DELTA_OUT:
        fused = ops.fused_delta_out(v, predicted, beta, q, k, q_decayed,
                                    v.shape[0], v.shape[-1], key=id(state))
    if fused is not None:
        delta, fused_out = fused
    else:
        fused_out = None
        delta = ttnn.multiply(ttnn.subtract(v, predicted), beta)
    # outer product kᵀ delta : [Dk, 1] x [1, Dv]. As a matmul this contracts over
    # a K of **one**, which is the worst shape a matmul has: 21.05 us against
    # 9.37 for the same numbers as a broadcast multiply, which is what an outer
    # product is. Both round to bfloat16 at the end and agree to 3.9e-03.
    kt = ttnn.transpose(k, -2, -1)
    # `update` is a full state-sized tensor -- 786 KB at batch 1 -- written once
    # and read once by the add below, and nothing else ever looks at it. One
    # kernel writing `state = decayed + kt (x) delta` removes 1.57 MB a layer,
    # 57 MB a token. No semaphores in it, so it cannot hang the way the k-split's
    # fold does.
    if not (not _NO_BCAST_OUTER and ops.fused_outer_add(decayed, kt, delta, state)):
        update = (ttnn.matmul(kt, delta, compute_kernel_config=HIFI4) if _NO_BCAST_OUTER
                  else ttnn.multiply(kt, delta))
        # write straight into the state buffer: `copy(add(...), state)` made two
        # full passes over it (1.23 ms vs 0.75 ms at B=32)
        ttnn.add(decayed, update, output_tensor=state)

    if fused_out is not None:
        return fused_out
    qk = ttnn.sum(ttnn.multiply(q, k), dim=-1, keepdim=True)          # [BH,1,1,1]
    return ttnn.add(q_decayed, ttnn.multiply(qk, delta))
