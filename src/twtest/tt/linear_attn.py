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

import ttnn

from .ops import HIFI4


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
    kq = ttnn.concat([k, q], dim=-2)                                  # [BH,1,2,Dk]
    both = ttnn.matmul(kq, decayed, compute_kernel_config=HIFI4)      # [BH,1,2,Dv]
    shape = list(both.shape)
    v_dim = shape[-1]
    predicted = ttnn.slice(both, (0, 0, 0, 0), (shape[0], shape[1], 1, v_dim))
    q_decayed = ttnn.slice(both, (0, 0, 1, 0), (shape[0], shape[1], 2, v_dim))

    delta = ttnn.multiply(ttnn.subtract(v, predicted), beta)
    # outer product kᵀ delta : [Dk, 1] x [1, Dv]
    update = ttnn.matmul(ttnn.transpose(k, -2, -1), delta, compute_kernel_config=HIFI4)
    # write straight into the state buffer: `copy(add(...), state)` made two full
    # passes over it (1.23 ms vs 0.75 ms at B=32)
    ttnn.add(decayed, update, output_tensor=state)

    qk = ttnn.sum(ttnn.multiply(q, k), dim=-1, keepdim=True)          # [BH,1,1,1]
    return ttnn.add(q_decayed, ttnn.multiply(qk, delta))
