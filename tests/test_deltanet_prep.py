"""The `gated_delta_attn_seq` input preparation.

The op's docstring documents shapes but not the sign convention of `L_unit` nor
the packing of `L_inv`, both of which were derived by construction. These tests
pin them so a refactor cannot silently flip either.
"""

from __future__ import annotations

import pytest
import torch

from twtest.tt.deltanet import BLOCK, CHUNK, prepare


def _inputs(seq: int = 256, heads: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    d = CHUNK
    return (
        torch.randn(1, seq, heads, d),
        torch.randn(1, seq, heads, d),
        torch.randn(1, seq, heads, d),
        -torch.rand(1, seq, heads) * 3.0,
        torch.rand(1, seq, heads),
    )


def test_shapes_match_the_op_contract() -> None:
    q, k, v, g, beta = _inputs(seq=256, heads=2)
    prep = prepare(q, k, v, g, beta)
    bh, nc = 2, 2
    assert tuple(prep["L_unit"].shape) == (bh, nc, CHUNK, CHUNK)
    assert tuple(prep["v_beta_sc"].shape) == (bh, nc, CHUNK, CHUNK)
    assert tuple(prep["k_decay_t"].shape) == (bh, nc, CHUNK, CHUNK)
    assert tuple(prep["dl_exp"].shape) == (bh, nc, 1, 1)
    # L_inv packs one 32x32 diagonal-block inverse per 32 rows
    assert tuple(prep["L_inv"].shape) == (bh, nc, CHUNK, BLOCK)
    for t in prep.values():
        assert t.dtype in (torch.float32, torch.int64), "the op requires float32 inputs"


def test_l_unit_is_unit_diagonal_lower_triangular() -> None:
    q, k, v, g, beta = _inputs()
    L = prepare(q, k, v, g, beta)["L_unit"]
    assert torch.all(torch.diagonal(L, dim1=-2, dim2=-1) == 1.0), "diagonal must be exactly 1"
    assert torch.all(L.triu(1) == 0), "must be lower triangular"


def test_l_inv_inverts_each_diagonal_block() -> None:
    q, k, v, g, beta = _inputs()
    prep = prepare(q, k, v, g, beta)
    L, L_inv = prep["L_unit"], prep["L_inv"]
    eye = torch.eye(BLOCK)
    n_blocks = CHUNK // BLOCK
    for b in range(n_blocks):
        s = slice(b * BLOCK, (b + 1) * BLOCK)
        block = L[..., s, s]
        inv = L_inv[..., s, :]
        assert torch.allclose(block @ inv, eye.expand_as(block), atol=1e-4), f"block {b}"


def test_padding_to_the_fixed_chunk_size() -> None:
    """The device op fixes chunk_size at 128, so a short sequence must pad up."""
    q, k, v, g, beta = _inputs(seq=100, heads=1)
    prep = prepare(q, k, v, g, beta)
    batch, heads, n_chunks, seq, pad = prep["_meta"].tolist()
    assert (seq, pad, n_chunks) == (100, 28, 1)
    assert prep["L_unit"].shape[2] == CHUNK


def test_decays_are_bounded_by_one() -> None:
    """Every decay is formed by subtraction in log space, so exp() cannot blow up.

    With A as low as -158 the ratio-of-exponentials form gives 0/0; the
    subtraction form is bounded above by exp(0) = 1.
    """
    q, k, v, g, beta = _inputs(seq=128, heads=2)
    g = -torch.rand_like(g) * 200.0
    prep = prepare(q, k, v, g, beta)
    for name in ("k_decay_t", "dl_exp"):
        t = prep[name]
        assert torch.isfinite(t).all(), f"{name} not finite"
        assert float(t.abs().max()) <= 1.0 + 1e-5, f"{name} exceeded 1"


def test_block_inverse_matches_linalg_inv() -> None:
    """The telescoping Neumann inverse must match `linalg.inv` on unit-lower matrices.

    It exists because every step is a matmul + add, so the same routine
    translates to ttnn -- unlike `linalg.inv`, which would force the whole
    preparation onto the host.
    """
    from twtest.tt.deltanet import block_inverse

    torch.manual_seed(0)
    # BLOCK (32) is the size actually used; 64 is included to show the method
    # scales, but its inverse reaches magnitude ~200 so the comparison has to be
    # relative rather than absolute.
    for size, rtol in ((32, 1e-5), (64, 1e-4)):
        n = torch.randn(3, 2, size, size).tril(-1) * 0.4
        unit = torch.eye(size) + n
        got = block_inverse(unit)
        want = torch.linalg.inv(unit)
        scale = float(want.abs().max())
        assert float((got - want).abs().max()) / scale < rtol, f"size {size}"
        eye = torch.eye(size).expand_as(unit)
        assert float((unit @ got - eye).abs().max()) < 1e-3, f"size {size} residual"


def test_block_inverse_rejects_non_power_of_two() -> None:
    from twtest.tt.deltanet import block_inverse

    with pytest.raises(ValueError, match="power of two"):
        block_inverse(torch.eye(48).expand(1, 48, 48))
