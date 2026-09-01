"""The chunked and recurrent gated delta rules must agree.

They are structurally different algorithms -- one a token loop, one a blocked
matmul formulation with a triangular inverse -- so agreement is real evidence.
"""

from __future__ import annotations

import pytest
import torch

from twtest.reference.layers import chunk_gated_delta_rule, recurrent_gated_delta_rule


def _inputs(seq: int, decay_scale: float, seed: int = 0):
    torch.manual_seed(seed)
    batch, heads, k_dim, v_dim = 2, 4, 32, 32
    return (
        torch.randn(batch, seq, heads, k_dim),
        torch.randn(batch, seq, heads, k_dim),
        torch.randn(batch, seq, heads, v_dim),
        -torch.rand(batch, seq, heads) * decay_scale,
        torch.rand(batch, seq, heads),
    )


@pytest.mark.parametrize("seq", [1, 7, 64, 100])
@pytest.mark.parametrize("decay_scale", [0.5, 160.0])
def test_chunked_matches_recurrent(seq: int, decay_scale: float) -> None:
    q, k, v, g, beta = _inputs(seq, decay_scale)
    out_rec, state_rec = recurrent_gated_delta_rule(q, k, v, g, beta)
    out_chunk, state_chunk = chunk_gated_delta_rule(q, k, v, g, beta)
    assert torch.allclose(out_rec, out_chunk, atol=1e-4), "outputs diverge"
    assert torch.allclose(state_rec, state_chunk, atol=1e-4), "carried state diverges"


def test_extreme_decay_keeps_state_finite() -> None:
    """Layer 0 has A = -158, so exp(cum_decay) underflows to exactly 0.

    Forming the position-to-chunk-end decay as exp(a)/exp(b) gives 0/0 = NaN in
    the *state* while leaving the chunk output finite -- it only surfaces on the
    next decode step. Guard against the regression.
    """
    q, k, v, g, beta = _inputs(seq=100, decay_scale=200.0)
    out, state = chunk_gated_delta_rule(q, k, v, g, beta)
    assert torch.isfinite(out).all(), "output went non-finite"
    assert torch.isfinite(state).all(), "carried recurrent state went non-finite"


def test_prefill_then_decode_matches_full_prefill() -> None:
    """Carrying the state across a chunked prefill and a recurrent decode step
    must equal prefilling the whole sequence at once."""
    seq = 33
    q, k, v, g, beta = _inputs(seq + 1, decay_scale=8.0)
    full, _ = chunk_gated_delta_rule(q, k, v, g, beta)

    pre = [x[:, :seq] for x in (q, k, v, g, beta)]
    _, state = chunk_gated_delta_rule(*pre)
    step = [x[:, seq : seq + 1] for x in (q, k, v, g, beta)]
    last, _ = recurrent_gated_delta_rule(*step, initial_state=state)
    assert torch.allclose(full[:, seq : seq + 1], last, atol=1e-4)
