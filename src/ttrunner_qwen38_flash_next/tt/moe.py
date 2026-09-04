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

import ttnn

from .ops import HIFI4

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
    logits = ttnn.linear(x, router_w, compute_kernel_config=HIFI4)
    probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=HIFI4)

    values, _ = ttnn.topk(probs, k=top_k, dim=-1, largest=True, sorted=True)
    v = list(values.shape)
    # the k-th largest probability is the inclusion threshold
    threshold = ttnn.slice(values, (0, 0, 0, top_k - 1), (v[0], v[1], v[2], top_k))
    # dtype is pinned so the mask multiplies cleanly against `probs` and the
    # sparsity cast below is a no-op rather than a conversion
    keep = ttnn.ge(probs, threshold, dtype=ttnn.bfloat16)   # [1, 1, M, E]
    kept = ttnn.multiply(probs, keep)
    weights = ttnn.divide(kept, ttnn.sum(kept, dim=-1, keepdim=True))

    # union of experts selected by any of the M rows -> [1, 1, 1, E]
    # sparse_matmul wants the mask rank-4, row-major and bfloat16
    sparsity = ttnn.max(keep, dim=-2, keepdim=True)
    sparsity = ttnn.to_layout(ttnn.typecast(sparsity, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)

    per_expert = expert_ffn(
        x, gate_w, up_w, down_w, sparsity, None, num_experts, hidden_size, intermediate_size
    )                                                     # [1, E, M, K]

    # [1, 1, M, E] -> [1, E, M, 1] so it broadcasts over the hidden axis
    return _combine(per_expert, weights, num_experts, hidden_size)


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
    logits = ttnn.linear(x, router_w, compute_kernel_config=HIFI4)
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

    sparsity = ttnn.to_layout(
        ttnn.typecast(ttnn.max(keep, dim=-2, keepdim=True), ttnn.bfloat16),
        ttnn.ROW_MAJOR_LAYOUT,
    )
    per_expert = expert_ffn(
        x, gate_w, up_w, down_w, sparsity, None, num_experts, hidden_size, intermediate_size
    )
    return _combine(per_expert, weights, num_experts, hidden_size)


def shared_expert(
    x: ttnn.Tensor,
    gate_w: ttnn.Tensor,
    up_w: ttnn.Tensor,
    down_w: ttnn.Tensor,
    gate_vec: ttnn.Tensor,
) -> ttnn.Tensor:
    """The always-on expert, with its own sigmoid gate."""
    hidden = ttnn.multiply(
        ttnn.silu(ttnn.linear(x, gate_w, compute_kernel_config=HIFI4)),
        ttnn.linear(x, up_w, compute_kernel_config=HIFI4),
    )
    out = ttnn.linear(hidden, down_w, compute_kernel_config=HIFI4)
    return ttnn.multiply(out, ttnn.sigmoid(ttnn.linear(x, gate_vec, compute_kernel_config=HIFI4)))


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
        out = ttnn.sparse_matmul(
            a, w, program_config=sparse_program_config(m, k_in, n_out),
            compute_kernel_config=HIFI4, **kw_in,
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
        e = both.shape[1]
        gate = ttnn.slice(both, (0, 0, 0, 0), (1, e, m, n))
        up = ttnn.slice(both, (0, 0, 0, n), (1, e, m, 2 * n))
    else:
        gate = _broadcast_matmul(gate_w, gate_w.shape[-1])
        up = _broadcast_matmul(up_w, up_w.shape[-1])
    hidden = ttnn.multiply(ttnn.silu(gate), up)
    pc_out = sparse_program_config(m, hidden.shape[-1], hidden_size)
    return ttnn.sparse_matmul(hidden, down_w, program_config=pc_out, compute_kernel_config=HIFI4, **kw)
