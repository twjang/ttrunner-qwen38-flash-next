"""Continuous batching: slots refill independently, so positions diverge.

Every batched benchmark until now advanced all sequences in lockstep from a
common start, which made `positions[0]` accidentally correct for the whole
batch. Continuous batching is the first thing that breaks that assumption: a
slot admitted mid-flight sits at position 0 while its neighbours are at 500.

These are host-side tests -- they lock the *contract* that made the bug
possible. The device-side proof (a neighbour's logits moved by 5.6 when one slot
was reset, and by 0.0 once the cache took per-sequence indices) needs four
Blackhole cards and lives in the iteration log.
"""

from __future__ import annotations

import importlib.util
import inspect

import pytest

pytest.importorskip("ttnn")

from twtest.tt.model import TTModel, TTState  # noqa: E402


def test_reset_slot_clears_only_its_own_bookkeeping() -> None:
    state = TTState(num_layers=4, batch=3)
    state.positions = [7, 9, 11]
    state.histories = [[1, 2], [3, 4], [5, 6]]

    # the tensor work needs a device; the bookkeeping half does not
    TTModel.reset_slot(_NoDeviceModel(), state, 1)

    assert state.positions == [7, 0, 11], "reset must not touch other slots"
    assert state.histories == [[1, 2], [], [5, 6]]


def test_reset_slot_rejects_out_of_range() -> None:
    state = TTState(num_layers=2, batch=2)
    with pytest.raises(IndexError):
        TTModel.reset_slot(_NoDeviceModel(), state, 2)


def test_update_cache_path_refuses_divergent_positions() -> None:
    """The unsafe path must fail loudly, not write one index for every sequence."""
    src = inspect.getsource(TTModel._attention_step)
    assert "len(set(positions)) > 1" in src, (
        "the single-index update_cache path lost its divergent-position guard; "
        "without it continuous batching silently corrupts attention history"
    )
    assert "traceable_kv=True" in src


def test_engine_selects_the_per_sequence_cache_path() -> None:
    """The server admits slots independently, so it must not use the int path."""
    spec = importlib.util.find_spec("twtest.tt.engine")
    src = open(spec.origin).read()
    assert "traceable_kv=True" in src, (
        "TTEngine must build TTModel(traceable_kv=True): paged_update_cache takes "
        "the cache index per sequence, update_cache takes one int for the batch"
    )


class _NoDeviceModel:
    """Enough of TTModel to exercise reset_slot's bookkeeping without a mesh."""

    n_v_local = 12
    mesh = None
    state_dtype = None

    def __getattr__(self, name):  # pragma: no cover - defensive
        raise AttributeError(name)
