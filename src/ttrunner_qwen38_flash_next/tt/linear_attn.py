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


def decode_step(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    g_exp: ttnn.Tensor,
    beta: ttnn.Tensor,
    state: ttnn.Tensor,
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
    update = (ttnn.matmul(kt, delta, compute_kernel_config=HIFI4) if _NO_BCAST_OUTER
              else ttnn.multiply(kt, delta))
    # write straight into the state buffer: `copy(add(...), state)` made two full
    # passes over it (1.23 ms vs 0.75 ms at B=32)
    ttnn.add(decayed, update, output_tensor=state)

    if fused_out is not None:
        return fused_out
    qk = ttnn.sum(ttnn.multiply(q, k), dim=-1, keepdim=True)          # [BH,1,1,1]
    return ttnn.add(q_decayed, ttnn.multiply(qk, delta))
