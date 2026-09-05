"""MoE expert FFN on device.

`ttnn.moe` is not a MoE layer (it returns expert-zero routing weights), so the
expert FFN is built on `ttnn.sparse_matmul`, whose
``(is_input_a_sparse, is_input_b_sparse) = (True, True)`` mode computes

    a [1, E, M, K] @ b [1, E, K, N] -> [1, E, M, N]

skipping every expert whose entry in ``sparsity [1, 1, 1, E]`` is zero.

The important consequence: **no token permutation is needed.** Broadcasting the
M tokens into all E expert slots and marking only the selected experts costs
``|selected| * M`` matmul rows. For decode (M = 1 per sequence) that is exactly
the 10 routed experts and nothing else -- optimal, with no data-dependent gather,
no host round-trip for a permutation, and no capacity padding. Prefill trades
some waste for the same simplicity and is therefore chunked.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import ttnn

from .ops import HIFI4, fast_linear, ksplit_linear

TILE = 32
_MM1D = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCast1DProgramConfig


def _tiles(n: int) -> int:
    return (n + TILE - 1) // TILE


def sparse_program_config(m: int, k: int, n: int, grid_x: int = 10, grid_y: int = 8) -> object:
    """Program config for `sparse_matmul`.

    Only the 1D multicast config is accepted, and `mcast_in0` must be True: the
    activation block is multicast to every core while the cores split N. With
    `per_core_N == 1` each core owns a single output tile column, which suits the
    narrow expert shapes here (N = 640 -> 20 tiles, N = 2560 -> 80 tiles).

    **`k` must be the input's real last dimension, not a nominal size.** It sets
    `in0_block_w`, and the op asserts `Kt % in0_block_w == 0` against the actual
    tensor.

    `in0_block_w` is the whole of K -- one block, no K loop -- and that is a
    correctness requirement, not a tuning choice. `ttnn.sparse_matmul` returns
    *wrong rows past the first 32-row tile* whenever `per_core_M > 1` and K
    spans more than one block, and it does so silently. Measured at
    E=64, N=320, M=64, varying only `in0_block_w` over a K of 80 tiles:

        in0_block_w   K blocks   rows wrong
                  8         10        32/64
                 16          5        32/64
                 40          2        16/64
                 80          1         0/64

    and by K at the default block width: K=256 (one block) clean, K=512 16/64,
    K>=1024 32/64. That is what made `moe_block` stop being per-token past 32
    rows and forced `_MAX_MOE_CHUNK`; see handoff 5.8 and
    `scripts/dev/sparse_matmul_rows_check.py`.

    The old candidate list (8, 5, 4, 2, 1) hid this for the down-projection,
    which it happened to give a single block by luck -- 5 candidates against a
    real Kt of 5 -- while the gate/up matmul got ten blocks and was wrong.
    """
    n_tiles, k_tiles, m_tiles = _tiles(n), _tiles(k), _tiles(m)
    cores = min(grid_x * grid_y, n_tiles)
    per_core_n = (n_tiles + cores - 1) // cores
    # One K block only where the bug can bite -- `per_core_M > 1`. At one row
    # tile the op is correct with any block width, and widening it there is not
    # free: it changes the accumulation order, and doing so unconditionally moved
    # prefill's NLL from 5.648 to 6.301 at the default `moe_chunk=32`, where
    # there was nothing to fix. Below the threshold this is the original
    # candidate list, so those numbers are untouched.
    in0_block_w = (
        k_tiles if m_tiles > 1
        else next((b for b in (8, 5, 4, 2, 1) if k_tiles % b == 0), 1)
    )
    return _MM1D(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=False,
        mcast_in0=True,
    )


def _localise(weights, keep, gate_w, num_experts: int):
    """Cut the routing down to the experts this device actually holds.

    The expert stacks are sharded on the expert axis, so a device has
    `gate_w.shape[1]` of the `num_experts` and needs exactly that slice of the
    mask. Every device computed the same `num_experts` logits -- the router is
    replicated, and softmax and top-k are global by definition -- so what is
    wanted is a partition of a replicated tensor, which is what
    `ttnn.mesh_partition` is: the inverse of all_gather, device 0 taking columns
    0..E/n and device 1 the next, matching how `ShardTensorToMesh` laid the
    weights out.

    Each device then produces a partial sum over its own experts, and the
    all-reduce the MoE already does adds them up.
    """
    e_local = gate_w.shape[1]
    if e_local == num_experts:
        return weights, keep, e_local
    return (ttnn.mesh_partition(weights, dim=-1),
            ttnn.mesh_partition(keep, dim=-1),
            e_local)


def _combine(per_expert, weights, num_experts: int, hidden_size: int):
    """Weighted sum over the expert axis: [1, E, M, K] x [1, 1, M, E] -> [1, 1, M, K].

    At M = 1 this is one matmul, and that is both the faster and the *more
    accurate* of the two forms.

    Faster because of the padding. TILE_LAYOUT pads the row axis to 32, so
    [1, 512, 1, 2560] holds 84 MB to carry 2.6 MB of answer; the elementwise form
    reads it, writes a product the same size and reads that back to reduce, about
    254 MB a layer. Reshaping the expert axis into the row axis first packs it to
    2.6 MB -- exact at M = 1, where each expert contributes one row -- and the
    weighted sum becomes a matmul against the router weights in the very shape
    routing produced them, so the permute goes too. 1.298 -> 0.481 ms a layer in
    isolation, 55.1 -> ~15 ms across 48 layers.

    More accurate because of where the rounding falls. The elementwise form
    rounds all E products to bfloat16 before adding them; the matmul accumulates
    them in fp32. Against the exact weighted sum in float64, computed from the
    device's own operands (`tiny_tile_check.py`):

        sum form     max abs 1.563e-03   mean abs 2.527e-04
        matmul form  max abs 1.138e-03   mean abs 1.650e-04

    So this is not a speed-for-accuracy trade; it is better on both counts. It is
    *not* bit-identical to what it replaces, which is why the evidence is here.

    Above M = 1 the packing would interleave (expert, row) and the gate would
    have to become a mostly-zero [M, E*M] matrix, so those callers keep the
    elementwise form -- where the row axis is full and the padding argument does
    not apply anyway.
    """
    m = per_expert.shape[-2]
    if m == 1:
        packed = ttnn.reshape(per_expert, (1, 1, num_experts, hidden_size))
        return ttnn.matmul(weights, packed, compute_kernel_config=HIFI4)
    gate_per_expert = ttnn.permute(weights, (0, 3, 2, 1))
    return ttnn.sum(ttnn.multiply(per_expert, gate_per_expert), dim=1, keepdim=True)


def moe_block(
    x: ttnn.Tensor,
    router_w: ttnn.Tensor,
    gate_w: ttnn.Tensor,
    up_w: ttnn.Tensor | None,
    down_w: ttnn.Tensor,
    top_k: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> ttnn.Tensor:
    """Routed-expert MoE, entirely on device.

    x: [1, 1, M, K] -> [1, 1, M, K] (a partial sum per device; the caller
    all-reduces because down_proj is sharded on its contraction dim).

    No host round-trip is needed anywhere in here, which matters because this
    runs 48 times per token:

    * `nnz=None` lets `sparse_matmul` count the non-zeros on device. Passing a
      count would mean reading the top-k indices back to the host, and a count
      that disagrees with the mask **hangs the device** (the compute kernel loops
      `nnz` times while the sender multicasts once per non-zero).
    * The routing mask is built by thresholding the probabilities at the k-th
      largest value rather than scattering indices, so no gather/scatter op and
      no index tensor ever leaves the device.

    Thresholding admits ties, so a row keeps ~11.5 experts where k is 10, and a
    row's set can move when bf16 rounding shifts a probability across the
    threshold. Exact top-k selection would fix the tie admission and *not* the
    movement: measured, the top-k set itself differs on the same 1 row in 64
    between a 64-row group and per-row calls, because the router's
    probabilities differ by up to 0.53 % under a different tiling and that
    genuinely reorders two experts across the k-th boundary. So scatter buys
    nothing here. See handoff 5.8.
    """
    # [2560, 512] is sixteen output tiles, so a 110-core grid affords six
    # reduction groups -- measured 1.49x and more accurate than `ttnn.linear`
    # (invariant 57). Six is about where the split starts paying; the guard in
    # `ksplit_linear` declines anything narrower in groups than that pays for.
    logits = ksplit_linear(x, router_w)
    if logits is None:
        logits = fast_linear(x, router_w, compute_kernel_config=HIFI4)
    probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=HIFI4)

    # Routing is global -- the router is replicated and top-k is over all of
    # `num_experts` -- but the expert stacks are sharded on the expert axis, so
    # from here on only this device's slice is wanted.
    #
    # **The wide path is exact at one row only.** It gathers a single set of
    # experts for the whole call, and `expert_gather.cpp` reads the selection
    # from the index tile's first face -- row 0. At M > 1 every row would then
    # be routed to row 0's experts. `scripts/dev/moe_rows_check.py` measures it:
    # with the wide path on, row groups of 8 get 55 of 64 rows wrong, worst row
    # 96.3 %, where group size 1 is exact. That is a real constraint and not a
    # tuning knob, so the guard is on M rather than on `_MAX_MOE_CHUNK`.
    # `step_n` (the speculative verifier) runs M = k, so it takes the
    # `sparse_matmul` path, which is per-row exact.
    if WIDE_EXPERTS and up_w is None:
        # Two gathers and two wide matmuls, the second of which also combines.
        # Falls back below if anything about the shapes is unexpected.
        # Not a silent fallback: swallowing the exception here would leave the
        # old path running while every measurement above claimed the new one, and
        # that is exactly the kind of thing that goes unnoticed. Report once.
        e_local = gate_w.shape[1]
        try:
            # One kernel for the whole routing tail: threshold, tie-admitting
            # mask, normalisation, this device's slice and its gather indices.
            # `sparsity` is not built at all -- the wide path never reads it,
            # and computing it was four discarded ops a layer.
            sel = fused_router_select(probs, num_experts, e_local, top_k, WIDE_EXPERTS)
            if sel is None:
                sel_weights = _route_weights(probs, top_k)
                w_local = (sel_weights if e_local == num_experts
                           else ttnn.mesh_partition(sel_weights, dim=-1))
            else:
                w_local = None
            rows = x.shape[-2]
            if rows == 1:
                return wide_expert_ffn(x, gate_w, down_w, w_local, WIDE_EXPERTS,
                                       hidden_size, sel=sel)
            # More than one row: the gather serves **one** selection, because
            # `expert_gather.cpp` reads it from the index tile's first face, so
            # each row has to have its own pass. That is `rows` times the M=1
            # cost -- and the alternative is `sparse_matmul` over all 512
            # experts, whose [1, E, M, K] zero-fill is a flat **145 ms** a step
            # whatever k is (`step_n_one.py --moe-stub`). Four rows of the wide
            # path is about 37.
            outs = []
            for i in range(rows):
                xi = ttnn.slice(x, (0, 0, i, 0), (1, 1, i + 1, x.shape[-1]))
                seli = None
                wi = w_local
                if sel is not None:
                    vals, idx = sel
                    seli = (ttnn.slice(vals, (0, 0, i, 0), (1, 1, i + 1, vals.shape[-1])),
                            ttnn.slice(idx, (0, 0, i, 0), (1, 1, i + 1, idx.shape[-1])))
                elif w_local is not None:
                    wi = ttnn.slice(w_local, (0, 0, i, 0),
                                    (1, 1, i + 1, w_local.shape[-1]))
                outs.append(wide_expert_ffn(xi, gate_w, down_w, wi, WIDE_EXPERTS,
                                            hidden_size, sel=seli))
            return ttnn.concat(outs, dim=-2)
        except Exception as exc:                            # noqa: BLE001
            global _WIDE_FELL_BACK
            if not _WIDE_FELL_BACK:
                _WIDE_FELL_BACK = True
                import warnings
                warnings.warn(
                    f"wide expert path unavailable, using sparse_matmul: "
                    f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

    values, _ = ttnn.topk(probs, k=top_k, dim=-1, largest=True, sorted=True)
    v = list(values.shape)
    # the k-th largest probability is the inclusion threshold
    threshold = ttnn.slice(values, (0, 0, 0, top_k - 1), (v[0], v[1], v[2], top_k))
    # dtype is pinned so the mask multiplies cleanly against `probs` and the
    # sparsity cast below is a no-op rather than a conversion
    keep = ttnn.ge(probs, threshold, dtype=ttnn.bfloat16)   # [1, 1, M, E]
    kept = ttnn.multiply(probs, keep)
    weights = ttnn.divide(kept, ttnn.sum(kept, dim=-1, keepdim=True))

    weights, keep, e_local = _localise(weights, keep, gate_w, num_experts)

    # union of experts selected by any of the M rows -> [1, 1, 1, E_local]
    # sparse_matmul wants the mask rank-4, row-major and bfloat16
    sparsity = ttnn.max(keep, dim=-2, keepdim=True)
    sparsity = ttnn.to_layout(ttnn.typecast(sparsity, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)

    per_expert = expert_ffn(
        x, gate_w, up_w, down_w, sparsity, None, e_local, hidden_size, intermediate_size
    )                                                     # [1, E_local, M, K]

    # a partial sum over this device's experts; the caller's all-reduce completes it
    return _combine(per_expert, weights, e_local, hidden_size)


def _route_weights(probs, top_k: int):
    """The threshold/mask/normalise tail, for when the fused kernel is not usable."""
    values, _ = ttnn.topk(probs, k=top_k, dim=-1, largest=True, sorted=True)
    v = list(values.shape)
    threshold = ttnn.slice(values, (0, 0, 0, top_k - 1), (v[0], v[1], v[2], top_k))
    keep = ttnn.ge(probs, threshold, dtype=ttnn.bfloat16)
    kept = ttnn.multiply(probs, keep)
    return ttnn.divide(kept, ttnn.sum(kept, dim=-1, keepdim=True))


def route(x, router_w, top_k: int):
    """The routing half of `moe_block`: probabilities in, (weights, keep) out.

    Split out so a caller can route in one row-group size and compute the
    experts in another. That is not a free choice -- routing at a different
    group size changes the answer (the router's probabilities move by up to
    0.53 % under a different tiling and that reorders experts across the k-th
    boundary; see `moe_block`) -- whereas the expert FFN is exactly per-token
    once `sparse_program_config` gives it a single K block. So `prefill` routes
    in 32-row groups, which is what `_MAX_MOE_CHUNK` pins, and computes the
    experts over the whole chunk.
    """
    logits = fast_linear(x, router_w, compute_kernel_config=HIFI4)
    probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=HIFI4)
    values, _ = ttnn.topk(probs, k=top_k, dim=-1, largest=True, sorted=True)
    v = list(values.shape)
    threshold = ttnn.slice(values, (0, 0, 0, top_k - 1), (v[0], v[1], v[2], top_k))
    keep = ttnn.ge(probs, threshold, dtype=ttnn.bfloat16)
    kept = ttnn.multiply(probs, keep)
    return ttnn.divide(kept, ttnn.sum(kept, dim=-1, keepdim=True)), keep


def apply_experts(x, weights, keep, gate_w, up_w, down_w, num_experts: int,
                  hidden_size: int, intermediate_size: int):
    """The compute half: run the union of selected experts over all of `x`.

    `x`, `weights` and `keep` all carry the same M rows; `weights`/`keep` may
    have been routed in smaller groups and concatenated.
    """
    m = x.shape[-2]
    if m > _MAX_FFN_ROWS:
        # `expert_ffn` materialises [1, E, M, K], so its footprint grows with the
        # row count times 512: at M=256 that is ~1.3 GB of intermediates and the
        # allocator gives up (program.cpp:1555). Running it in row groups is free
        # of consequence -- the FFN is exactly per-token (`expert_split_check.py`
        # measures max diff 0.000e+00 against the whole-chunk form), and each
        # group's `sparsity` is the union over its *own* rows, so an expert a
        # group does not select is one it also weights at zero. Same answer,
        # bounded memory, and it is what lets a prefill chunk be wider than the
        # DeltaNet op's 128.
        outs = []
        for lo in range(0, m, _MAX_FFN_ROWS):
            hi = min(lo + _MAX_FFN_ROWS, m)
            k = list(x.shape)
            xs = ttnn.slice(x, (0, 0, lo, 0), (k[0], k[1], hi, k[3]))
            w = list(weights.shape)
            ws = ttnn.slice(weights, (0, 0, lo, 0), (w[0], w[1], hi, w[3]))
            ks = ttnn.slice(keep, (0, 0, lo, 0), (w[0], w[1], hi, w[3]))
            outs.append(apply_experts(xs, ws, ks, gate_w, up_w, down_w,
                                      num_experts, hidden_size, intermediate_size))
        return ttnn.concat(outs, dim=-2)

    weights, keep, e_local = _localise(weights, keep, gate_w, num_experts)
    sparsity = ttnn.to_layout(
        ttnn.typecast(ttnn.max(keep, dim=-2, keepdim=True), ttnn.bfloat16),
        ttnn.ROW_MAJOR_LAYOUT,
    )
    per_expert = expert_ffn(
        x, gate_w, up_w, down_w, sparsity, None, e_local, hidden_size, intermediate_size
    )
    return _combine(per_expert, weights, e_local, hidden_size)


# --- fused SwiGLU ------------------------------------------------------------
#
# `silu(gate) * up` over the [1, E, M, 2N] gate|up matmul output used to be four
# ttnn ops -- two slices, a silu, a multiply -- and that is not op overhead: the
# four together move ~30 MB a layer, because each reads and writes the whole
# E=128 tensor, and at 388 GB/s that is the 87 us measured. One fused pass reads
# both halves once and writes the result once, and measures **40.05 us against
# 87.06, 2.17x, 2.26 ms a token** (`scripts/stage3_swiglu_fusion.py`).
#
# `ttnn.swiglu` does not do this: it is a composite that issues the same four ops
# (invariant 43). This is three real kernels under `ttnn.generic_op`, with the
# intermediate living in circular buffers and never reaching DRAM.
_KDIR = Path(__file__).resolve().parents[3] / "scripts" / "kernels"
_SWIGLU_OUT: dict = {}
_SWIGLU_FELL_BACK = False
# The shared expert had its own unfused copy of the SwiGLU chain.
_NO_SHEXP_SWIGLU = bool(os.environ.get("TT_NO_SHEXP_SWIGLU"))


def _swiglu_output(src, n_out: int):
    """The persistent output buffer, one per (device, shape, dtype).

    `generic_op` needs its output pre-allocated, and allocating inside a trace
    capture corrupts the replay -- so this is made once and reused by all 48
    layers. That is safe because a layer's hidden is consumed by its own down
    projection before the next layer runs, and a trace replays in order.
    """
    key = (id(src.device()), src.shape[1], src.shape[2], n_out, str(src.dtype))
    out = _SWIGLU_OUT.get(key)
    if out is None:
        out = ttnn.from_torch(
            torch.zeros(1, src.shape[1], src.shape[2], n_out),
            dtype=src.dtype, layout=ttnn.TILE_LAYOUT, device=src.device(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(src.device()),
        )
        _SWIGLU_OUT[key] = out
    return out


def _swiglu_program(src, out):
    """Reader/compute/writer descriptors for this call's buffer addresses.

    Rebuilt per call rather than cached: `src` is a fresh allocation out of the
    matmul, so its address is part of the runtime args. The cost is host-side and
    is paid at trace capture, not at replay -- the recorded commands carry the
    addresses, and a replay puts the tensors back at the same places.
    """
    grid = src.device().compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )
    acc_i = list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
    acc_o = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    if len(acc_i) != 2 or acc_o[1] != acc_i[1]:
        raise RuntimeError("fused SwiGLU wants interleaved tensors of one dtype")
    tile_bytes = acc_i[1]

    nt_out = out.shape[-1] // TILE
    rows = out.shape[1] * max(out.shape[2] // TILE, 1)
    total = rows * nt_out
    work = [((total * c) // len(cores), (total * (c + 1)) // len(cores))
            for c in range(len(cores))]

    cbs = [ttnn.CBDescriptor(
        total_size=4 * tile_bytes, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=src.dtype, page_size=tile_bytes)])
        for i in (0, 1, 2)]

    def kern(name, ct, args, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, a) for c, a in zip(cores, args)], config=cfg)

    return ttnn.ProgramDescriptor(
        kernels=[
            kern("swiglu_reader.cpp", [nt_out, tile_bytes] + acc_i,
                 [[src.buffer_address(), lo, hi] for lo, hi in work],
                 ttnn.ReaderConfigDescriptor()),
            kern("swiglu_compute.cpp", [], [[hi - lo] for lo, hi in work],
                 ttnn.ComputeConfigDescriptor()),
            kern("swiglu_writer.cpp", [tile_bytes] + acc_o,
                 [[out.buffer_address(), lo, hi] for lo, hi in work],
                 ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[], cbs=cbs)


def fused_swiglu(both, n_out: int):
    """silu(first half) * second half, in one pass. Falls back to the ops."""
    try:
        out = _swiglu_output(both, n_out)
        ttnn.generic_op([both, out], _swiglu_program(both, out))
        return out
    except Exception as exc:                                        # noqa: BLE001
        global _SWIGLU_FELL_BACK
        if not _SWIGLU_FELL_BACK:
            _SWIGLU_FELL_BACK = True
            import warnings
            warnings.warn(f"fused swiglu unavailable, using the ops: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        e, m = both.shape[1], both.shape[2]
        gate = ttnn.slice(both, (0, 0, 0, 0), (1, e, m, n_out))
        up = ttnn.slice(both, (0, 0, 0, n_out), (1, e, m, 2 * n_out))
        return ttnn.multiply(ttnn.silu(gate), up)


# --- the wide-gather expert path ---------------------------------------------
#
# `sparse_matmul` is good at the matmul (invariant 44) but everything around it
# pays for an expert axis of 128 when ~3 experts are wanted: a mask-independent
# zero-fill of 1.51 GB a token (invariant 40) and every downstream elementwise op
# sized by E rather than by the selection (invariant 42).
#
# Gathering the selected experts side by side into one wide matrix removes all of
# it. The layout is what decides whether that is worth doing: gathered into an
# expert *batch* it loses to sparse_matmul, gathered into width it beats it 7x
# (invariant 52). Measured at the safe K_SEL=10, gate/up is 10.60 ms over 48
# layers against 30.83 for both projections today.
#
# The down projection is gathered with the other concatenation, on its input
# axis, so its matmul sums over the experts -- which is `_combine`, for free.
_GATHER_KERNEL = str(Path(__file__).resolve().parents[3] / "scripts" / "kernels"
                     / "expert_gather.cpp")
_GATHER_BUF: dict = {}
_IDX_LEN = 128            # index page width in uint32 -> 512 B
# Tiles in flight between barriers in the expert gather. The gather is a pure
# copy at 47 % of bandwidth and this is its only knob; swept, and eight is the
# floor -- 4, 8, 16 and 32 measure 98.8, 97.4, 97.7 and 98.6 us. It is not
# read-depth-bound, so making it faster means not copying at all.
_READ_BATCH = int(os.environ.get("TT_GATHER_BATCH", "8"))
# Selection width for the wide path, or 0 to keep `sparse_matmul`.
#
# With the experts sharded on the *intermediate* axis every device holds all of
# them, so this is a global count rather than a per-device worst case: the same
# k_sel experts run on all four devices, each over a quarter of their columns.
# `kept_expert_census.py` measures what it has to cover -- the router admits
# ties, so a row keeps 10 experts 77 % of the time and up to 14 over 2256 real
# routing decisions -- and 16 is both above that tail and the gather kernel's own
# limit, since the selection is read from one tile face.
#
# Sixteen slots at a quarter width is 2.5x less work than the ten full-width
# slots the expert-axis layout needed, and unlike those ten it wastes nothing:
# every slot but the tie tail carries a real expert.
#
# Except that the tie tail is not wanted. `reference/model.py` selects with
# `torch.topk(probs, num_experts_per_tok)` and normalises over exactly those --
# **no tie admission**. This project's threshold form (`probs >= the top_k-th
# value`) is the approximation, and it admits ~1.5 extra experts a row as an
# artifact of comparing in bfloat16. The expert-axis shard absorbed that for
# free, because an extra landed on whichever device already owned it; with every
# device holding every expert it has to be paid for in slots, 16 rather than 10,
# which is 1.6x the gather and both matmuls.
#
# Measured on 234 tokens, the two rules are indistinguishable -- **top-5
# identical at 200/234 and NLL 1.514 against 1.515** -- while top-1 differs by 3
# tokens, which is 0.4 sigma. So the exact rule is the reference's own, the same
# quality, and 62.31 -> 58.42 ms a token. `TT_ROUTER_EXACT_TOPK=0` restores the
# threshold.
_NO_SHEXP_KSPLIT = bool(os.environ.get("TT_NO_SHEXP_KSPLIT"))
ROUTER_EXACT_TOPK = os.environ.get("TT_ROUTER_EXACT_TOPK", "1") == "1"
WIDE_EXPERTS = 10 if ROUTER_EXACT_TOPK else 16
_WIDE_FELL_BACK = False


def _buf(key, shape, dtype, device):
    """A persistent output for `generic_op`, which cannot allocate under capture."""
    got = _GATHER_BUF.get(key)
    if got is None:
        got = ttnn.from_torch(
            torch.zeros(*shape), dtype=dtype, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device))
        _GATHER_BUF[key] = got
    return got


def _gather_program(weights, indices, out, k_sel: int, wide: int, idx16: int = 0):
    grid = weights.device().compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    acc = {}
    for tag, t in (("w", weights), ("i", indices), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"{tag}: the gather wants interleaved tensors")
        acc[tag] = ct
    tile_bytes, idx_bytes = acc["w"][1], acc["i"][1]

    kt, nt = weights.shape[-2] // TILE, weights.shape[-1] // TILE
    tpe = kt * nt
    total = k_sel * tpe
    work = [((total * c) // len(cores), (total * (c + 1)) // len(cores))
            for c in range(len(cores))]

    cbs = [
        ttnn.CBDescriptor(
            total_size=(_READ_BATCH + 1) * tile_bytes, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=0, data_format=weights.dtype, page_size=tile_bytes)]),
        ttnn.CBDescriptor(
            total_size=64 * ((idx_bytes + 64 + 63) // 64), core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=1, data_format=ttnn.uint32, page_size=64)]),
    ]
    ct_args = [tpe, tile_bytes, _READ_BATCH, idx_bytes,
               weights.shape[1], k_sel, nt, wide, idx16]
    ct_args += acc["w"] + acc["i"] + acc["o"]
    addrs = (weights.buffer_address(), indices.buffer_address(), out.buffer_address())
    kernel = ttnn.KernelDescriptor(
        kernel_source=_GATHER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs, compile_time_args=ct_args,
        runtime_args=[(c, [*addrs, lo, hi]) for c, (lo, hi) in zip(cores, work)],
        config=ttnn.ReaderConfigDescriptor())
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)


def _gather(weights, idx_u32, k_sel: int, wide: int, out_shape, idx16: int = 0):
    out = _buf(("g", wide, out_shape, str(weights.dtype)), out_shape,
               weights.dtype, weights.device())
    ttnn.generic_op([weights, idx_u32, out],
                    _gather_program(weights, idx_u32, out, k_sel, wide, idx16))
    return out

# --- the routing tail as one kernel ------------------------------------------
#
# `decode_ablation_check.py` prices the chain this replaces, against an 82.11 ms
# step: the global `ttnn.topk` over 512 experts is 5.00 ms a token, the
# threshold/mask/normalise 0.93, `mesh_partition` 0.73, and the local `topk`
# below another 2.00 -- 8.66 ms of sorting 512 numbers. Removing four of the
# chain's ops moved the step 0.24 ms, so this is not op-count overhead; it is
# those two sorts, and only a kernel gets at them.
#
# `scripts/stage7_router_select.py` measures 151.59 -> 27.18 us, **5.58x**, and
# checks the result three ways: against float64, against the ttnn chain, and --
# the one that matters -- against the *effective per-expert weight vector*, max
# diff 3.05e-05. Slot order can differ where two experts tie, which bf16
# probabilities do often; summing weight x expert makes that difference vanish,
# which is exactly what the gather and the matmul downstream do.
_ROUTER_KERNEL = str(Path(__file__).resolve().parents[3] / "scripts" / "kernels"
                     / "router_select.cpp")
_ROUTER_BUF: dict = {}
_ROUTER_FELL_BACK = False


def _router_devid(device, partitioned: bool):
    """Which expert window this device should select from, as a tensor.

    `generic_op` broadcasts one program to the whole mesh, so runtime args are
    identical everywhere and cannot say which device is running. A tensor can:
    sharded on dim 0, each device gets its own page. Sixteen uint32 rather than
    one, so the page clears DRAM's 64 B alignment.

    `partitioned` is False when every device holds every expert -- which is what
    the intermediate-axis shard gives -- and then all four select the same global
    list, so the window index is zero everywhere.
    """
    key = ("devid", id(device), partitioned)
    got = _ROUTER_BUF.get(key)
    if got is None:
        n = device.get_num_devices()
        ids = (torch.arange(n, dtype=torch.int32) if partitioned
               else torch.zeros(n, dtype=torch.int32))
        got = ttnn.from_torch(
            ids.reshape(n, 1, 1, 1).expand(n, 1, 1, 16).contiguous(),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0))
        _ROUTER_BUF[key] = got
    return got


def _router_program(probs, devid, vals, idx, m_rows, e_total, e_local, top_k, k_sel):
    dev = probs.device()
    core = ttnn.CoreCoord(0, 0)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    acc = {}
    for tag, t in (("p", probs), ("d", devid), ("v", vals), ("i", idx)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"router select: {tag} must be interleaved")
        acc[tag] = ct
    if probs.dtype == ttnn.bfloat16:
        p_bf16 = 1
    elif probs.dtype == ttnn.float32:
        p_bf16 = 0
    else:
        raise RuntimeError(f"router select: probs dtype {probs.dtype} unsupported")
    v_bf16 = 1 if vals.dtype == ttnn.bfloat16 else 0

    # A CB's total size must be a whole number of its page size.
    probs_cb = acc["p"][1] * (e_total // TILE + 2)
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
    ct = [e_total, e_local, top_k, k_sel, m_rows, p_bf16, v_bf16,
          acc["p"][1], acc["v"][1], acc["i"][1], int(ROUTER_EXACT_TOPK)]
    ct += acc["p"] + acc["d"] + acc["v"] + acc["i"]
    kernel = ttnn.KernelDescriptor(
        kernel_source=_ROUTER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs, compile_time_args=ct,
        runtime_args=[(core, [probs.buffer_address(), devid.buffer_address(),
                              vals.buffer_address(), idx.buffer_address()])],
        config=ttnn.ReaderConfigDescriptor())
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)


def fused_router_select(probs, e_total: int, e_local: int, top_k: int, k_sel: int):
    """(weights, local expert ids) for the gather, in one launch. None if unusable.

    Returns exactly what `topk(mesh_partition(normalise(threshold(probs))))`
    returns, in the layouts `wide_expert_ffn` already consumes: bfloat16 weights
    [1, 1, M, k_sel] and uint16 ids in a tile's first face, which is what
    `expert_gather.cpp` reads under IDX16=1.
    """
    global _ROUTER_FELL_BACK
    dev = probs.device()
    m_rows = probs.shape[-2]
    try:
        # An off switch, so the kernel and the chain it replaces can be compared
        # in one sitting rather than across a code change. Both are the same
        # distance from float64 (`stage7_router_select.py`), so a difference in
        # generated text is the tie order and not accuracy -- but that is a
        # claim worth being able to re-measure.
        if os.environ.get("TT_NO_FUSED_ROUTER"):
            return None
        if m_rows > TILE or k_sel > 16 or top_k > TILE:
            return None
        key = (id(dev), m_rows, k_sel, str(probs.dtype))
        got = _ROUTER_BUF.get(key)
        if got is None:
            rep = ttnn.ReplicateTensorToMesh(dev)
            got = (
                ttnn.from_torch(torch.zeros(1, 1, m_rows, k_sel), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep),
                ttnn.from_torch(torch.zeros(1, 1, m_rows, k_sel), dtype=ttnn.uint16,
                                layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep),
            )
            _ROUTER_BUF[key] = got
        vals, idx = got
        devid = _router_devid(dev, e_local != e_total)
        ttnn.generic_op(
            [probs, devid, vals, idx],
            _router_program(probs, devid, vals, idx, m_rows, e_total, e_local,
                            top_k, k_sel))
        return vals, idx
    except Exception as exc:                                    # noqa: BLE001
        if not _ROUTER_FELL_BACK:
            _ROUTER_FELL_BACK = True
            import warnings
            warnings.warn(
                f"fused router select unavailable, using the ttnn chain: "
                f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


def wide_expert_ffn(x, gate_w, down_w, weights_local, k_sel, hidden_size,
                    sel=None):
    """The whole MoE for this device: two gathers, two wide matmuls, one SwiGLU.

    `weights_local` is the router's normalised weights restricted to this
    device's experts, zero where an expert was not selected. `ttnn.topk` on it
    hands back both the local ids to gather and the scores to scale by, and the
    zero-scored padding entries contribute nothing -- so a fixed `k_sel` is safe
    as long as it is the worst case, which for top-10 over four devices is 10.

    Shapes, with N the intermediate width and E the local expert count:

        gather gate|up (WIDE=2)   [1, 1, K, k_sel*2N]   gates then ups
        linear                     [1, 1, M, k_sel*2N]
        fused SwiGLU               [1, 1, M, k_sel*N]
        scale by the router        (one small matmul broadcasts k_sel -> k_sel*N)
        gather down (WIDE=0)       [1, 1, k_sel*N, K]   experts stacked on rows
        linear                     [1, 1, M, K]         <- sums the experts

    That last matmul is the combine: stacking the down slabs on their input axis
    makes the contraction run over experts as well as over N. The caller's
    all-reduce then completes the sum across devices, exactly as before.
    """
    dev = x.device()
    n = gate_w.shape[-1] // 2                     # intermediate width
    if sel is not None:
        # `fused_router_select` produced both in one launch, replacing this
        # topk and the whole chain that fed it.
        vals, idx = sel
    else:
        vals, idx = ttnn.topk(weights_local, k=k_sel, dim=-1, largest=True, sorted=True)

    # `topk` hands back uint16 in TILE layout, and the kernel reads that tile's
    # first face directly -- columns 0..15 are contiguous there, so k_sel <= 16
    # needs no typecast, no layout change and no pad. Three ops a layer, 144 a
    # token, gone. Validated against a torch gather rather than assumed.
    idx_pad = idx
    if k_sel > 16:
        raise ValueError(f"k_sel {k_sel} exceeds a tile face; widen the index path")

    gu = _gather(gate_w, idx_pad, k_sel, 2, (1, 1, gate_w.shape[-2], k_sel * 2 * n), 1)
    both = fast_linear(x, gu, compute_kernel_config=HIFI4)
    hidden = fused_swiglu(both, k_sel * n)

    # Broadcast each expert's score across its n columns. `repeat_interleave` is
    # pathological here (14.4 ms, see `reinject`), so it is one small matmul
    # against a constant 0/1 block matrix instead.
    spread = _buf(("spread", k_sel, n), (1, 1, k_sel, k_sel * n), ttnn.bfloat16, dev)
    if _GATHER_BUF.get(("spread_init", k_sel, n)) is None:
        blk = torch.zeros(1, 1, k_sel, k_sel * n)
        for sidx in range(k_sel):
            blk[0, 0, sidx, sidx * n:(sidx + 1) * n] = 1.0
        ttnn.copy(ttnn.from_torch(blk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=dev,
                                  mesh_mapper=ttnn.ReplicateTensorToMesh(dev)), spread)
        _GATHER_BUF[("spread_init", k_sel, n)] = True
    scaled = ttnn.multiply(hidden, ttnn.matmul(vals, spread, compute_kernel_config=HIFI4))

    dw = _gather(down_w, idx_pad, k_sel, 0, (1, 1, k_sel * n, hidden_size), 1)
    return fast_linear(scaled, dw, compute_kernel_config=HIFI4)


_SHEXP_GU: dict = {}


def shared_expert(
    x: ttnn.Tensor,
    gate_w: ttnn.Tensor,
    up_w: ttnn.Tensor,
    down_w: ttnn.Tensor,
    gate_vec: ttnn.Tensor,
) -> ttnn.Tensor:
    """The always-on expert, with its own sigmoid gate."""
    # gate and up take the same input, so they are one matmul with a wider
    # output: 3.48 ms a token becomes 1.93 (invariant 55).
    key = (id(gate_w), id(up_w))
    fused = _SHEXP_GU.get(key)
    if fused is None:
        fused = ttnn.concat([gate_w, up_w], dim=-1)
        _SHEXP_GU[key] = fused
    both = fast_linear(x, fused, compute_kernel_config=HIFI4)
    n = gate_w.shape[-1]
    e, mrows = both.shape[1], both.shape[2]
    # The same fusion `expert_ffn` has used since stage 3, which this path never
    # picked up: two slices, a silu and a multiply, 48 times a token, where the
    # kernel keeps the halves in circular buffers and never writes them out.
    if _NO_SHEXP_SWIGLU:
        hidden = ttnn.multiply(
            ttnn.silu(ttnn.slice(both, (0, 0, 0, 0), (1, e, mrows, n))),
            ttnn.slice(both, (0, 0, 0, n), (1, e, mrows, 2 * n)),
        )
    else:
        hidden = fused_swiglu(both, n)
    out = fast_linear(hidden, down_w, compute_kernel_config=HIFI4)
    # The sigmoid gate is a single output column, and a single column is one
    # output tile on one core of a hundred and ten: measured 22.13 us a call for
    # a 10 KB weight, **0.1 % of bandwidth**, 48 calls a token. It is the exact
    # shape the k-split exists for -- give the other 79 cores a slice of the
    # 2560-long reduction instead.
    gate = None if _NO_SHEXP_KSPLIT else ksplit_linear(x, gate_vec)
    if gate is None:
        gate = fast_linear(x, gate_vec, compute_kernel_config=HIFI4)
    return ttnn.multiply(out, ttnn.sigmoid(gate))


# Above this many rows the per-expert input copy is cheaper than making the op
# broadcast one set of rows to every expert. 64 is the largest width measured to
# win (2.35x); 128 loses in the model. See `expert_ffn`.
_BROADCAST_MAX_M = 64

# Row groups for the expert FFN. Only bites above a 128-row prefill chunk, so the
# default path is untouched; see `apply_experts`.
_MAX_FFN_ROWS = 128


def expert_ffn(
    x: ttnn.Tensor,
    gate_w: ttnn.Tensor,
    up_w: ttnn.Tensor,
    down_w: ttnn.Tensor,
    sparsity: ttnn.Tensor,
    nnz: int | None,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> ttnn.Tensor:
    """Run the selected experts over all M rows of `x`.

    x: [1, 1, M, K];  gate_w/up_w: [1, E, K, N];  down_w: [1, E, N, K]
    returns [1, E, M, K]
    """
    m = x.shape[-2]
    kw = {"sparsity": sparsity, "nnz": nnz, "is_input_a_sparse": True, "is_input_b_sparse": True}

    # The gate/up matmuls take the *same* rows for every expert, so below
    # `_BROADCAST_MAX_M` rows they use the op's (dense a, sparse b) mode -- a is
    # [1, 1, M, K] against [1, E, K, N], output [1, 1, 1, E, M, N] -- instead of
    # handing it E copies of x.
    #
    # At decode that copy is nearly all padding: M is 1, TILE_LAYOUT pads the row
    # axis to 32, so `repeat(x, (1, E, 1, 1))` materialised [1, 512, 32, 2560] to
    # carry 512 copies of a single row -- ~84 MB a layer, 48 layers a token, one
    # row in 32 of it real. Dropping it takes the traced step 266.3 -> 238.4 ms.
    #
    # It reverses at a full prefill chunk, which is why this is a threshold and
    # not a replacement. Measured in isolation at E=512 the broadcast mode wins
    # 1.94x at M=1, 1.82x at M=32 and 2.35x at M=64; in the model at M=128 it
    # *loses*, 919.5 -> 930.0 ms a chunk, where the rows are real data rather
    # than padding and the op has to fan them out to every expert instead. Both
    # forms are bit-identical (`sparse_broadcast_check.py` compares every
    # element at each M), so this picks purely on measured speed.
    #
    # Only these two can use it either way. The down projection's `a` is the
    # per-expert intermediate, which genuinely differs per expert.
    kw_in = dict(kw)
    if m <= _BROADCAST_MAX_M:
        kw_in["is_input_a_sparse"] = False

    def _broadcast_matmul(w, n_out):
        """gate/up for every expert. -> [1, E, M, N]

        In broadcast mode the op returns [1, 1, 1, E, M, N]; the reshape drops
        leading unit dimensions only, so it is metadata.
        """
        a = x if kw_in["is_input_a_sparse"] is False else ttnn.repeat(x, (1, num_experts, 1, 1))
        # `dtype=bfloat8_b` on the gate/up output. It sets two costs at once:
        # `sparse_matmul` zero-fills its whole [1, E, M, N] output on every call
        # regardless of the mask (handoff 5.2, 1.51 GB a token), and the SwiGLU
        # chain then reads that output over all E. Halving the element width
        # halves both.
        #
        # The precision case: this tensor is the product of bfloat4_b weights,
        # so its low bits are already noise, and it feeds silu and a multiply
        # rather than an accumulation. Kept only because the quality harness
        # held -- see the commit; if it had moved, this comes straight back out.
        out = ttnn.sparse_matmul(
            a, w, program_config=sparse_program_config(m, k_in, n_out),
            compute_kernel_config=HIFI4, dtype=ttnn.bfloat8_b, **kw_in,
        )
        return ttnn.reshape(out, (1, num_experts, m, n_out))

    # Both configs come from the tensors, not from the nominal sizes. The expert
    # weights are sharded, so this device's intermediate width is
    # `gate_w.shape[-1] // 2`, a quarter of `intermediate_size` -- passing the
    # nominal value asks for an `in0_block_w` the op rejects outright
    # ("Kt (5) must be divisible by in0_block_w (20)").
    k_in = x.shape[-1]

    if up_w is None:
        # gate_w carries gate|up fused on the output axis: one sparse_matmul
        # instead of two. Measured 7.749 -> 4.210 ms at M=64 (1.84x) on the
        # gate+up pair, because the cost here is per-call, not per-element.
        # The fused tensor must come from scripts/fuse_expert_gate_up.py, which
        # quantises the concatenation once; concatenating the quantised halves on
        # device requantises them and changes the output.
        n = gate_w.shape[-1] // 2
        both = _broadcast_matmul(gate_w, 2 * n)
        hidden = fused_swiglu(both, n)
    else:
        gate = _broadcast_matmul(gate_w, gate_w.shape[-1])
        up = _broadcast_matmul(up_w, up_w.shape[-1])
        hidden = ttnn.multiply(ttnn.silu(gate), up)
    pc_out = sparse_program_config(m, hidden.shape[-1], hidden_size)
    # Deliberately NOT `dtype=bfloat8_b` here, unlike the gate/up call above.
    # It was tried: 101.69 -> 101.45 ms, which is inside the run-to-run spread,
    # because `hidden` is already bfloat8_b so this matmul's input had already
    # halved and only the output fill was left. Quality was unchanged, but this
    # tensor is what `_combine` weights and sums into the layer's answer rather
    # than an intermediate, and 0.24 ms is not a reason to spend precision on
    # the output path.
    return ttnn.sparse_matmul(hidden, down_w, program_config=pc_out, compute_kernel_config=HIFI4, **kw)
