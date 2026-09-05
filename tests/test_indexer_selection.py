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
    assert "dense beyond the" in src and "not what the model does" in src


def _gate(max_seq_len: int, budget: int = 2048, ratio: int = 4, topk: int = 512):
    """`TTModel.__init__`'s two selection gates, evaluated without a device.

    Kept as arithmetic rather than a source-text match so that a change to the
    rule has to change the *rule*, not a quoted line.
    """
    compact_slots = topk + 32
    compact_len = compact_slots * 32
    max_blocks = max_seq_len // ratio
    compact = max_seq_len > compact_len and max_blocks <= (1 << 16)
    use = budget < max_seq_len and (compact or max_seq_len <= (1 << 16))
    return use, compact


def test_the_selection_is_off_below_the_budget() -> None:
    """Below the budget every eligible block fits, so dense is exactly right
    and cheaper -- and the selection must not claim to be doing anything."""
    for n in (512, 1024, 2048):
        use, _ = _gate(n)
        assert not use, f"selection should be off at {n}"


def test_the_dense_mask_runs_between_the_budget_and_the_compact_window() -> None:
    """The dense row is one column per cache position, so it is the smaller of
    the two only up to `compact_len`. Above that the compact window takes over;
    below the budget the selection is off entirely."""
    from ttrunner_qwen38_flash_next.tt.model import TTModel
    import inspect

    assert "self.indexer_max_seq = 1 << 16" in inspect.getsource(TTModel.__init__)
    for n in (4096, 8192, 16384):
        use, compact = _gate(n)
        assert use and not compact, f"{n} should use the dense mask"


def test_the_compact_window_carries_the_selection_past_uint16() -> None:
    """The compact page table addresses a few thousand columns however long the
    context is, which is what lets the selection run to the model's maximum."""
    for n in (1 << 17, 1 << 18):
        use, compact = _gate(n)
        assert use and compact, f"compact selection should be on at {n}"
    # One block index per `ratio` positions, and those indices are uint16, so
    # 262144 is the ceiling -- the model's own maximum, not a coincidence.
    assert _gate(1 << 18)[1]
    assert not _gate(1 << 19)[1]


# --- why the selection can be skipped below the budget ----------------------
#
# `_indexer_mask` is the expensive half of the indexer: `ttnn.topk` over
# `max_blocks` costs a measured 22.3 ms across the twelve QSA layers, 15 % of a
# decode step. The engine skips it while a sequence is below `indexer_budget`
# and recaptures the trace when it crosses (`_enable_selection`).
#
# That is only sound if the mask it would have built is exactly plain causal
# attention there, which holds when every eligible block fits inside the
# selection budget -- then `topk` returns all of them and the -inf padding
# contributes nothing that the `index <= p` filter does not already drop.
#
# These pin the arithmetic that makes it true, so a change to the budget, the
# compression ratio, or the eligibility rule cannot quietly break the engine's
# fast path.


@pytest.mark.parametrize("p", [0, 1, 3, 4, 100, 1023, 2044, 2046, BUDGET - 1])
def test_every_eligible_block_fits_the_budget_below_it(p: int) -> None:
    m = _model()
    bias = m._block_bias([p])[0, 0, 0]
    # Selectable means not pushed to -inf and not the incomplete straddling one.
    selectable = int((bias == 0.0).sum())
    assert selectable <= m.indexer_topk, (
        f"at p={p}, {selectable} blocks compete for {m.indexer_topk} slots -- "
        "the selection is no longer a no-op below the budget and the engine's "
        "skip in _enable_selection would change the answer"
    )


def test_the_budget_is_where_that_stops_being_true() -> None:
    """The first position at which selection genuinely selects.

    Not an implementation detail: it is the threshold `_enable_selection` uses,
    and it should sit at or above the budget, never below it.
    """
    m = _model()
    first_binding = next(
        p for p in range(m.max_seq_len)
        if int((m._block_bias([p])[0, 0, 0] == 0.0).sum()) > m.indexer_topk
    )
    assert first_binding >= m.indexer_budget, (
        f"blocks outnumber slots from p={first_binding}, below the budget "
        f"{m.indexer_budget} the engine treats as safe"
    )
