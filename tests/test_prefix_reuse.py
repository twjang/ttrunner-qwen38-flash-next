"""Prefix reuse: a follow-up chat turn should not re-feed the whole history.

A slot that a finished sequence left behind still holds that conversation in
its recurrent state, its convolution rings and its K/V cache. The next turn of
the same chat contains the previous turn as an exact prefix, so it only has to
feed what it added -- at ~0.23 s per token on the step path, a 2000-token
history is 7.7 minutes of work that does not have to happen twice.

What makes it safe is knowing exactly which tokens a slot's state consumed. A
slot with no sequence is still fed a filler token on every step so the batch
keeps its shape, and that moves its state on; the bookkeeping has to give up on
such a slot rather than claim a prefix it no longer holds.

Host-side: these lock the decision, not the device state.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("ttnn")

from ttrunner_qwen38_flash_next.tt.engine import TTEngine, reusable_prefix  # noqa: E402


def test_exact_prefix_with_a_token_left_over_is_reusable() -> None:
    assert reusable_prefix([1, 2, 3], [1, 2, 3, 4])
    assert reusable_prefix([], [1])


def test_unrelated_or_diverging_prompts_are_not_reusable() -> None:
    assert not reusable_prefix([1, 2, 3], [1, 2, 9, 4])
    assert not reusable_prefix([1, 2, 3], [9])
    # a longer held sequence is not a prefix of a shorter prompt
    assert not reusable_prefix([1, 2, 3, 4], [1, 2, 3])


def test_a_prompt_equal_to_the_held_prefix_starts_over() -> None:
    """There would be no token left to feed, and the step that consumes the
    last token is the one that produces the first output logits."""
    assert not reusable_prefix([1, 2, 3], [1, 2, 3])


def test_an_unaccounted_slot_is_never_reused() -> None:
    assert not reusable_prefix(None, [1, 2, 3])


def test_admission_resets_only_when_the_prefix_does_not_carry() -> None:
    src = inspect.getsource(TTEngine._device_loop)
    admit = src[src.index("def admit("):]
    admit = admit[: admit.index("\n        while not self._shutdown")]
    assert "reusable_prefix(held, prompt)" in admit
    assert "seq.prompt_pos = len(held)" in admit
    # the reset lives on the branch that could not reuse
    assert admit.index("else:") < admit.index("self.model.reset_slot(state, slot)")


def test_slots_fed_filler_stop_claiming_a_prefix() -> None:
    src = inspect.getsource(TTEngine._device_loop)
    body = src[src.index("# The step happened"):]
    assert "prefix[i] = None" in body
    assert "prefix[i].append(tokens[i])" in body


def test_a_device_fault_invalidates_every_slot() -> None:
    src = inspect.getsource(TTEngine._device_loop)
    fault = src[src.index("a device fault kills every slot"):]
    assert "prefix[:] = [None] * B" in fault[:400]
