"""QSA's block selection must not depend on how the prompt was fed.

The reference is the oracle for the whole project, and it disagreed with itself:
a prompt run through one forward gave a different next token than the same
prompt fed token by token, diverging at the first sparse-attention layer.

The cause was the "trailing partial block is always visible" rule. A query at
position p may only use blocks that end at or before p, so everything after the
last such block has to come from the tail -- which is a property of p, not of
the sequence. Taking it from the global `kv_len` gave every query but the last a
tail belonging to someone else, and when `kv_len` was an exact multiple of the
block size it gave them no tail at all: a fully masked row.

These run on plain torch -- no checkpoint, no device.
"""

from __future__ import annotations

import torch

from ttrunner_qwen38_flash_next.reference.model import Qwen4ExpModel

HIDDEN, DIM, HEADS, RATIO, BUDGET = 8, 4, 2, 4, 16


class _Cfg:
    indexer_head_dim = DIM
    indexer_compress_ratio = RATIO
    indexer_budget = BUDGET
    rms_norm_eps = 1e-6


class _Store:
    def __init__(self, weights):
        self._w = weights

    def get(self, name):
        return self._w[name.split(".", 2)[2]]


def _model(seed: int = 0) -> Qwen4ExpModel:
    g = torch.Generator().manual_seed(seed)
    weights = {
        "indexer.q_proj.weight": torch.randn(HEADS * DIM, HIDDEN, generator=g),
        "indexer.k_proj.weight": torch.randn(DIM, HIDDEN, generator=g),
        "indexer.q_norm.weight": torch.ones(DIM),
        "indexer.k_norm.weight": torch.ones(DIM),
    }
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    model.config = _Cfg()
    model.store = _Store(weights)
    return model


def _mask(model, hidden, kv_len, query_offset):
    seq = hidden.shape[1]
    rope = torch.zeros(1, max(kv_len, seq), DIM)
    return model._indexer_mask(hidden, 0, rope + 1.0, rope, None, kv_len, query_offset)


def test_a_whole_prompt_selects_what_each_query_would_select_alone() -> None:
    """Row p of a whole-sequence call must equal the last row of a call that
    stops at p -- which is what the decode path computes, one token at a time."""
    model = _model()
    g = torch.Generator().manual_seed(1)
    for total in (4, 5, 8, 12, 16):
        hidden = torch.randn(1, total, HIDDEN, generator=g)
        whole = _mask(model, hidden, total, 0)[0, 0]              # (total, total)
        for p in range(total):
            alone = _mask(model, hidden[:, : p + 1], p + 1, 0)[0, 0, -1]
            assert torch.equal(whole[p, : p + 1], alone), (total, p)


def test_every_query_can_see_at_least_itself() -> None:
    """A fully masked row makes the softmax degenerate. It happened whenever
    `kv_len` was an exact multiple of the block size."""
    model = _model()
    g = torch.Generator().manual_seed(2)
    for total in (4, 8, 12, 16, 20):
        hidden = torch.randn(1, total, HIDDEN, generator=g)
        mask = _mask(model, hidden, total, 0)[0, 0]
        assert mask.any(dim=-1).all(), f"a query sees nothing at kv_len={total}"
        # Once there are more complete blocks than the budget keeps, selection
        # may drop the block a query sits in -- that is the design, and the row
        # is still non-empty. What must never happen is a row with nothing in it.


def test_nothing_beyond_a_query_is_visible() -> None:
    model = _model()
    g = torch.Generator().manual_seed(3)
    hidden = torch.randn(1, 13, HIDDEN, generator=g)
    mask = _mask(model, hidden, 13, 0)[0, 0]
    future = torch.triu(torch.ones(13, 13, dtype=torch.bool), diagonal=1)
    assert not (mask & future).any()


def test_below_the_budget_everything_causal_is_visible() -> None:
    """The device path attends densely and calls that exact below the budget.
    With `budget // ratio` blocks available, selection keeps them all."""
    model = _model()
    g = torch.Generator().manual_seed(4)
    total = BUDGET          # 16 tokens = 4 blocks = exactly block_topk
    hidden = torch.randn(1, total, HIDDEN, generator=g)
    mask = _mask(model, hidden, total, 0)[0, 0]
    causal = ~torch.triu(torch.ones(total, total, dtype=torch.bool), diagonal=1)
    assert torch.equal(mask, causal)
