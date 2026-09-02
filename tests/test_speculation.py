"""Prompt-lookup drafting and greedy acceptance.

Speculation is only worth having if its output is *identical* to not
speculating, so the accept rule is the load-bearing part: stop at the first
disagreement, keep the prefix, and take the model's own token at the boundary.

Measured with the rollback and the replay of the accepted prefix both paid for
(`scripts/dev/ngram_accept_rate.py`): 1.70-2.14x on prompts that quote their
context, 0.95-1.00x on open prose. The drafter fires on 36-67 % of positions in
the first case and 3-9 % in the second, which is the whole story.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ttnn")

from twtest.tt.engine import accepted_prefix, prompt_lookup_draft  # noqa: E402


def test_draft_is_what_followed_the_last_earlier_occurrence() -> None:
    ctx = [1, 2, 3, 9, 9, 1, 2, 3]
    assert prompt_lookup_draft(ctx, k=2, ngram=3) == [9, 9]


def test_the_most_recent_occurrence_wins() -> None:
    ctx = [1, 2, 3, 7, 7, 5, 5, 1, 2, 3, 8, 8, 5, 5, 1, 2, 3]
    assert prompt_lookup_draft(ctx, k=2, ngram=3) == [8, 8]


def test_no_draft_when_the_ngram_never_recurred() -> None:
    assert prompt_lookup_draft([1, 2, 3, 4, 5], k=2, ngram=3) is None


def test_a_match_too_close_to_the_end_does_not_end_the_search() -> None:
    """The nearest match may have fewer than k tokens after it. Giving up there
    silently costs drafts; the search has to continue further back."""
    # the nearest match (index 7) leaves only three tokens; the one at index 0
    # leaves four, so that is the draft
    ctx = [1, 2, 3, 5, 6, 7, 8, 1, 2, 3, 1, 2, 3]
    assert prompt_lookup_draft(ctx, k=4, ngram=3) == [5, 6, 7, 8]
    assert prompt_lookup_draft(ctx, k=3, ngram=3) == [1, 2, 3]  # the nearer one suffices


def test_no_draft_when_no_occurrence_has_k_tokens_after_it() -> None:
    assert prompt_lookup_draft([1, 2, 3, 0, 1, 2, 3], k=5, ngram=3) is None


def test_no_draft_from_a_context_shorter_than_the_ngram() -> None:
    assert prompt_lookup_draft([1, 2], k=2, ngram=3) is None
    assert prompt_lookup_draft([1, 2, 3], k=2, ngram=3) is None


def test_a_draft_never_overlaps_its_own_key() -> None:
    """Matching the tail itself would propose the tail back, which is a draft of
    tokens the model has already emitted."""
    ctx = [5, 5, 5, 5, 5, 5]
    got = prompt_lookup_draft(ctx, k=2, ngram=3)
    assert got == [5, 5]  # a real earlier occurrence, not the tail


def test_acceptance_stops_at_the_first_disagreement() -> None:
    assert accepted_prefix([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert accepted_prefix([1, 2, 3, 4], [9, 2, 3, 4]) == 0
    assert accepted_prefix([1, 2, 3, 4], [1, 2, 3, 4]) == 4


def test_acceptance_is_bounded_by_whichever_is_shorter() -> None:
    assert accepted_prefix([1, 2, 3], [1, 2]) == 2
    assert accepted_prefix([], [1, 2]) == 0


def test_the_engine_refuses_speculation_it_cannot_serve() -> None:
    import inspect

    from twtest.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.__init__)
    # batched decoding, an untraced verifier, and the sparse selection
    assert "max_concurrency=1 (got {max_concurrency})" in src
    assert "beaten by the path it replaces" in src
    assert "step_n does not carry the QSA selection yet" in src


def test_there_is_exactly_one_capture() -> None:
    """Capturing a second `step_n` graph allocates its intermediates while the
    first trace is live, which tt-metal warns corrupts them."""
    import inspect

    from twtest.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.__init__)
    assert "self._widths = [speculate] if speculate else []" in src
    assert "(len(self._widths) + 1) * (192 << 20)" in src, "the region is sized for it"
    loop = inspect.getsource(TTEngine._device_loop)
    # the replay still composes, from whatever widths exist plus single steps
    assert "max((w for w in verifiers if w <= remaining), default=1)" in loop


def test_speculation_snapshots_before_it_verifies() -> None:
    """A recurrence cannot be truncated back to the accepted prefix the way a
    K/V cache can, so a rejected draft has to be rolled back."""
    import inspect

    from twtest.tt.engine import TTEngine

    src = inspect.getsource(TTEngine._device_loop)
    block = src[src.index("def speculate_round("):]
    assert block.index("self.model.snapshot(state)") < block.index("step_n(feed)")
    assert "self.model.restore(state, snap)" in block
