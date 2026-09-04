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

from ttrunner_qwen38_flash_next.tt.engine import accepted_prefix, prompt_lookup_draft  # noqa: E402


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


def test_the_engine_records_that_speculation_is_exact() -> None:
    """Greedy acceptance *does* make this exact, contrary to what was recorded.

    `scripts/dev/speculation_exactness_check.py` drives the round loop -- draft,
    snapshot, verify, accept, restore, replay -- against a plain sequential
    decode from the same start, and the token streams are identical at k=2, 8
    and 17 over 64 tokens. The old argument, that `step_n` must round
    differently for computing row i in a k-row batch, has a false premise:
    `step_n` reproduces k sequential steps exactly to k=32, because k <= 32 rows
    and 1 row are both inside one row tile (handoff invariant 13).
    """
    import inspect

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.__init__)
    assert "**Identical** to decoding one token at a time" in src, (
        "the engine must record that the scheme is exact -- verified by "
        "speculation_exactness_check.py at k=2, 8 and 17"
    )


def test_speculation_refuses_rather_than_wedging_the_device() -> None:
    """Replaying the two traces alternately hangs, and the boards then need
    `tt-smi -r`. A flag that bricks the accelerators is worse than no flag, so
    it raises rather than tries -- unless a developer opts in explicitly."""
    import inspect
    import os

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.__init__)
    assert "raise NotImplementedError" in src
    assert "replayed" in src and "alternately" in src, (
        "the refusal must name what actually hangs -- the *replay* of the "
        "verifier, not its capture; see docs/iterations/018"
    )
    assert "TWTEST_ALLOW_SPECULATION" in src, "there must be a way to work on it"
    # and the escape must be opt-in, not merely present
    assert not os.environ.get("TWTEST_ALLOW_SPECULATION"), (
        "this test asserts the default; unset TWTEST_ALLOW_SPECULATION to run it"
    )


def test_the_second_command_queue_is_not_reintroduced() -> None:
    """It stops the hang and returns wrong tokens, which is a worse trade.

    On its own queue the step_n replay does not execute -- it merely stops
    blocking -- returning [201058, 0] where the eager `step_n` returns
    [75, 220], and coming back in 12 ms against 265. It showed up first as the
    engine accepting 0 of every 5 drafted tokens, and it was briefly committed
    as the fix on the evidence that the hang had stopped. See
    `docs/iterations/018` and `scripts/dev/spec_capture_ladder.py`.
    """
    import inspect

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.__init__)
    # Positive anchor: this is an "X is absent" test, which a gutted or renamed
    # `__init__` would satisfy for the wrong reason.
    assert "open_mesh_device" in src and "trace_region_size" in src, (
        "must be reading the real mesh setup"
    )
    assert "num_command_queues" not in src, (
        "a second command queue trades the hang for a silently wrong replay"
    )


def test_acceptance_stops_at_the_first_disagreement() -> None:
    assert accepted_prefix([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert accepted_prefix([1, 2, 3, 4], [9, 2, 3, 4]) == 0
    assert accepted_prefix([1, 2, 3, 4], [1, 2, 3, 4]) == 4


def test_acceptance_is_bounded_by_whichever_is_shorter() -> None:
    assert accepted_prefix([1, 2, 3], [1, 2]) == 2
    assert accepted_prefix([], [1, 2]) == 0


def test_the_engine_refuses_speculation_it_cannot_serve() -> None:
    import inspect

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine

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

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.__init__)
    assert "drafts\n        # `speculate - 1`" in src or "speculate - 1" in src
    assert "2 <= speculate <= 17" in src
    # the decode trace stays on: ordinary rounds want it
    assert "and not self._speculate" not in src


def test_the_verifier_is_eager_and_needs_no_capture() -> None:
    """The verifier must not be a second trace, and the replay needs no ladder.

    Two traces replayed alternately hang this build (`ttnn_bug_report/`), and a
    speculating engine alternates by construction -- the decoder on plain
    rounds, the verifier on drafted ones. Running the verifier eagerly leaves
    the decoder's capture as the only trace, never alternated against anything.

    It also removes the width ladder the captured version needed: with nothing
    to compile, any width is free, so a partial acceptance replays in one
    `step_n(j+1)` rather than composing powers of two.
    """
    import inspect

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine

    loop = inspect.getsource(TTEngine._device_loop)
    # Comments stripped: the code explains *why* the verifier is eager by naming
    # the captured class it replaced, and matching raw source would make that
    # explanation fail the test it exists for. Same precedent as
    # `test_prefill_contract._code_of`.
    code = "\n".join(line.split("#", 1)[0] for line in loop.split("\n"))
    assert "TracedStepN" not in code, "the verifier must not be captured"
    assert "self.model.step_n(feed, state)" in code, "it must verify eagerly"
    assert "self.model.step_n(feed[: j + 1], state)" in code, "and replay in one call"
    assert "self.model.step_n([0] * width, state)" in code, (
        "every width must be warmed before the decoder is captured: allocating "
        "a step_n graph while a trace is live hangs the device"
    )


def test_speculation_snapshots_before_it_verifies() -> None:
    """A recurrence cannot be truncated back to the accepted prefix the way a
    K/V cache can, so a rejected draft has to be rolled back."""
    import inspect

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine

    src = inspect.getsource(TTEngine._device_loop)
    block = src[src.index("def speculate_round("):]
    assert block.index("self.model.snapshot(state") < block.index("step_n(feed, state)")
    assert "self.model.restore(state, snap)" in block
    # and the buffers are reused, not reallocated every round: ~200 allocations
    # per round is slow, and allocating while a trace is live is the hazard
    assert "into=snap_buf.get(" in block


def test_close_releases_every_capture_before_closing_the_mesh() -> None:
    """Closing the mesh with a trace still registered does not fail there -- it
    fails in whatever opens the devices next, and as a hang rather than an error.
    Two engines in one process is enough to hit it."""
    import inspect

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine

    src = inspect.getsource(TTEngine.close)
    assert "trace.release()" in src
    assert src.index("trace.release()") < src.index("close_mesh_device")
    assert "self._verifiers" in src, (
        "close must still sweep them: the verifier is eager now, but `close` is "
        "what stopped a second engine in one process hanging on its own capture"
    )


def test_snapshot_buffers_are_allocated_before_any_capture() -> None:
    """A round snapshots ~200 tensors. Allocating them for the first time after
    a trace exists is the hazard `traced.py` opens by describing -- and it is the
    one thing the standalone harnesses never do, which is why they never hang."""
    import inspect

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine

    src = inspect.getsource(TTEngine._device_loop)
    alloc = src.index('snap_buf["s"] = self.model.snapshot(state)')
    assert alloc < src.index("TracedDecoder"), "must precede the decoder capture"
