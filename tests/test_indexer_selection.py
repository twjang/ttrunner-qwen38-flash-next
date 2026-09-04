"""The device's QSA selection rules, as plain arithmetic.

`_block_bias` and `_tail_block` are what decide which of a 262144-token context
a query may attend to, and both are pure functions of the position -- which is
what lets the selection run inside a captured trace, since a trace replays one
graph and cannot branch on p.

Two rules, and getting either wrong is quiet:

* A block is selectable once all `ratio` of its tokens are visible. The block p
  sits inside, while incomplete, is pushed *below* the other ineligible ones so
  `topk` can never return it -- it is appended separately, because spending a
  selection slot on it would keep 511 blocks where the reference keeps 512 (this
  was measured against the reference: 2047 tokens visible against 2051).
* Once that block completes it is eligible like any other, competes in `topk`,
  and nothing is appended -- appending it then would force in a block the
  reference only allows if it scores.

Host-side: no device, no checkpoint.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ttnn")

from ttrunner_qwen38_flash_next.tt.model import TTModel  # noqa: E402

RATIO, BUDGET, MAX_SEQ, K_CHUNK = 4, 2048, 4096, 128


def _model() -> TTModel:
    m = TTModel.__new__(TTModel)
    m.indexer_ratio = RATIO
    m.indexer_budget = BUDGET
    m.indexer_topk = BUDGET // RATIO
    m.max_blocks = MAX_SEQ // RATIO
    m.sdpa_k_chunk = K_CHUNK
    m.indexer_window = BUDGET + K_CHUNK
    m.max_seq_len = MAX_SEQ
    return m


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 7, 8, 100, 2047, 2048, 2599, 2598])
def test_only_complete_blocks_are_selectable(p: int) -> None:
    bias = _model()._block_bias([p])[0, 0, 0]
    for j in range(MAX_SEQ // RATIO):
        complete = RATIO * j + RATIO - 1 <= p
        straddling = j == p // RATIO and p % RATIO != RATIO - 1
        if straddling:
            assert bias[j] < -1e9, f"block {j} must be unselectable at p={p}"
        elif complete:
            assert bias[j] == 0.0, f"block {j} is complete at p={p}"
        else:
            assert bias[j] == -1e9, f"block {j} is incomplete at p={p}"


@pytest.mark.parametrize("p", [0, 1, 2, 4, 5, 100, 2598])
def test_an_incomplete_block_is_appended_with_its_visible_prefix(p: int) -> None:
    idx, vis = _model()._tail_block([p])
    start = RATIO * (p // RATIO)
    for t in range(RATIO):
        assert idx[0, 0, 0, t] == start + t
        assert vis[0, 0, 0, t] == (1.0 if start + t <= p else 0.0), (p, t)
    assert (vis[0, 0, 0, RATIO:] == 0.0).all(), "padding must not be visible"


@pytest.mark.parametrize("p", [3, 7, 2047, 2599])
def test_a_complete_block_is_not_appended(p: int) -> None:
    """It competes in topk instead; appending would force it in."""
    _, vis = _model()._tail_block([p])
    assert (vis == 0.0).all(), f"nothing should be appended at p={p}"


@pytest.mark.parametrize("p", [0, 1, 2, 3, 100, 2598, 2599])
def test_padding_slots_point_somewhere_never_visible(p: int) -> None:
    """They are scattered with a zero, so pointing at a token some selected
    block legitimately contributed would un-select it. Index 0 would do exactly
    that: block 0 may be selected and token 0 is visible from the first step."""
    idx, vis = _model()._tail_block([p])
    for t in range(K_CHUNK):
        if vis[0, 0, 0, t] == 0.0:
            assert int(idx[0, 0, 0, t]) > p or t < RATIO, (p, t)
    pad = [int(idx[0, 0, 0, t]) for t in range(RATIO, K_CHUNK)]
    assert all(x > p for x in pad), "padding must be beyond the current position"


def test_the_appended_chunk_never_duplicates_a_selectable_block() -> None:
    """The two paths are disjoint, so no key is attended to twice -- which would
    silently double its softmax weight."""
    m = _model()
    for p in range(0, 300):
        bias = m._block_bias([p])[0, 0, 0]
        idx, vis = m._tail_block([p])
        appended = {
            int(idx[0, 0, 0, t]) // RATIO for t in range(K_CHUNK) if vis[0, 0, 0, t] == 1.0
        }
        for j in appended:
            assert bias[j] < -1e9, f"block {j} is both appended and selectable at p={p}"


def test_every_position_can_see_itself() -> None:
    m = _model()
    for p in range(0, 300):
        bias = m._block_bias([p])[0, 0, 0]
        idx, vis = m._tail_block([p])
        visible = {int(idx[0, 0, 0, t]) for t in range(K_CHUNK) if vis[0, 0, 0, t] == 1.0}
        own_block_selectable = bias[p // RATIO] == 0.0
        assert p in visible or own_block_selectable, f"p={p} cannot see itself"


def test_positions_are_handled_independently() -> None:
    """Continuous batching puts sequences at different positions in one step."""
    m = _model()
    bias = m._block_bias([3, 100, 2598])
    assert bias.shape == (3, 1, 1, MAX_SEQ // RATIO)
    for row, p in enumerate([3, 100, 2598]):
        assert torch.equal(bias[row], m._block_bias([p])[0])


def test_the_engine_says_when_the_selection_cannot_run() -> None:
    """Dense attention past the budget is a different model, so it must not be
    something a caller discovers from the output."""
    import inspect

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.__init__)
    assert "not self.model.use_indexer and seq > self.config.indexer_budget" in src
    assert "attention is" in src and "dense beyond the budget" in src


def test_the_selection_is_off_where_it_cannot_address_the_cache() -> None:
    """`ttnn.scatter` takes uint16 indices, so 65536 cache positions is the
    reach; and below the budget dense is exactly right and cheaper."""
    import inspect

    from ttrunner_qwen38_flash_next.tt.model import TTModel

    src = inspect.getsource(TTModel.__init__)
    assert "config.indexer_budget < max_seq_len <= self.indexer_max_seq" in src
    assert "self.indexer_max_seq = 1 << 16" in src
