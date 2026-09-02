"""Prompt-lookup drafting and greedy acceptance.

The accept rule is the load-bearing part: stop at the first disagreement, keep
the prefix, and take the model's own token at the boundary. That makes every
emitted token the argmax of the verifier's own logits -- though *not* the same
stream as stepping one token at a time, because the verifier batches k rows
where the stepper runs one and bf16 rounding differs in the last bits.

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


def test_the_engine_does_not_promise_identical_output() -> None:
    """Greedy acceptance is exact only if the verifier and the stepper agree bit
    for bit. `step_n` batches k rows where `step` runs one, bf16 rounding
    differs, and argmax amplifies it -- measured divergence at token 29 and 39.
    Every emitted token is still the argmax of the verifier's own logits."""
    import inspect

    from twtest.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.__init__)
    assert "**Not** identical to decoding one token at a time" in src
    assert "speculation is experimental" in src


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
    # batched decoding has no single sequence to verify
    assert "max_concurrency=1 (got {max_concurrency})" in src
    # step_n has no sparse selection, so it would silently disagree past 2048
    assert "step_n does not carry the QSA selection yet" in src


def test_speculate_counts_tokens_fed_not_tokens_drafted() -> None:
    """It drafts `speculate - 1`. Reading it the other way put the engine in its
    worst configuration: at speculate=2 the draft is one token, the drafter fires
    on nearly every round, and each failure pays a verify plus a replay -- 492 ms
    a token against a 240 ms baseline."""
    import inspect

    from twtest.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.__init__)
    assert "drafts\n        # `speculate - 1`" in src or "speculate - 1" in src
    assert "2 <= speculate <= 17" in src
    # the decode trace stays on: ordinary rounds want it
    assert "and not self._speculate" not in src


def test_the_replay_widths_are_a_ladder() -> None:
    """The verify needs `speculate`; a partial acceptance replays 1..speculate-1.
    Powers of two plus the verify width cover any prefix by composition, and a
    second live capture was measured not to slow the first one's replay."""
    import inspect

    from twtest.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.__init__)
    assert "sorted({w for w in (2, 4, 8, 16) if w < speculate}" in src
    assert "(len(self._widths) + 1) * (128 << 20)" in src
    loop = inspect.getsource(TTEngine._device_loop)
    assert "max((w for w in verifiers if w <= remaining), default=1)" in loop


def test_speculation_snapshots_before_it_verifies() -> None:
    """A recurrence cannot be truncated back to the accepted prefix the way a
    K/V cache can, so a rejected draft has to be rolled back."""
    import inspect

    from twtest.tt.engine import TTEngine

    src = inspect.getsource(TTEngine._device_loop)
    block = src[src.index("def speculate_round("):]
    assert block.index("self.model.snapshot(state") < block.index("step_n(feed)")
    assert "self.model.restore(state, snap)" in block
    # and the buffers are reused, not reallocated every round: ~200 allocations
    # per round is slow, and allocating while a trace is live is the hazard
    assert "into=snap_buf.get(" in block


def test_close_releases_every_capture_before_closing_the_mesh() -> None:
    """Closing the mesh with a trace still registered does not fail there -- it
    fails in whatever opens the devices next, and as a hang rather than an error.
    Two engines in one process is enough to hit it."""
    import inspect

    from twtest.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.close)
    assert "trace.release()" in src
    assert src.index("trace.release()") < src.index("close_mesh_device")
    assert "self._verifiers" in src
    # and the engine keeps hold of them so close can find them
    assert "self._verifiers = verifiers" in inspect.getsource(TTEngine._device_loop)
