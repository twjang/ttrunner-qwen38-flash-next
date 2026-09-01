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
    """
    n_tiles, k_tiles, m_tiles = _tiles(n), _tiles(k), _tiles(m)
    cores = min(grid_x * grid_y, n_tiles)
    per_core_n = (n_tiles + cores - 1) // cores
    # widest block that still divides K, capped so the L1 block stays small
    in0_block_w = next((b for b in (8, 5, 4, 2, 1) if k_tiles % b == 0), 1)
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
    gate_per_expert = ttnn.permute(weights, (0, 3, 2, 1))
    return ttnn.sum(ttnn.multiply(per_expert, gate_per_expert), dim=1, keepdim=True)


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
    broadcast = ttnn.repeat(x, (1, num_experts, 1, 1))
    kw = {"sparsity": sparsity, "nnz": nnz, "is_input_a_sparse": True, "is_input_b_sparse": True}

    pc_in = sparse_program_config(m, hidden_size, intermediate_size)
    pc_out = sparse_program_config(m, intermediate_size, hidden_size)

    if up_w is None:
        # gate_w carries gate|up fused on the output axis: one sparse_matmul
        # instead of two. Measured 7.749 -> 4.210 ms at M=64 (1.84x) on the
        # gate+up pair, because the cost here is per-call, not per-element.
        # The fused tensor must come from scripts/fuse_expert_gate_up.py, which
        # quantises the concatenation once; concatenating the quantised halves on
        # device requantises them and changes the output.
        n = gate_w.shape[-1] // 2
        both = ttnn.sparse_matmul(
            broadcast, gate_w, program_config=sparse_program_config(m, hidden_size, 2 * n),
            compute_kernel_config=HIFI4, **kw,
        )
        e = both.shape[1]
        gate = ttnn.slice(both, (0, 0, 0, 0), (1, e, m, n))
        up = ttnn.slice(both, (0, 0, 0, n), (1, e, m, 2 * n))
    else:
        gate = ttnn.sparse_matmul(broadcast, gate_w, program_config=pc_in, compute_kernel_config=HIFI4, **kw)
        up = ttnn.sparse_matmul(broadcast, up_w, program_config=pc_in, compute_kernel_config=HIFI4, **kw)
    hidden = ttnn.multiply(ttnn.silu(gate), up)
    return ttnn.sparse_matmul(hidden, down_w, program_config=pc_out, compute_kernel_config=HIFI4, **kw)
