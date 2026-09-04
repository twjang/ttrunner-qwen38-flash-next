"""The decoding state's shape and bookkeeping.

Token histories moved from per-layer to per-sequence when the model was
generalised to batched decode; these lock that so callers (and harnesses) do not
drift back to the old per-layer field.
"""

from __future__ import annotations

import importlib.util

import pytest

# TTState lives in tt.model, which imports ttnn at module scope.
pytest.importorskip("ttnn")

from ttrunner_qwen38_flash_next.tt.model import LayerState, TTState  # noqa: E402


def test_state_is_per_sequence_not_per_layer() -> None:
    state = TTState(num_layers=48, batch=3)
    assert len(state.layers) == 48
    assert state.batch == 3
    assert state.positions == [0, 0, 0]
    assert state.histories == [[], [], []]
    # histories are per sequence; a layer must not carry its own copy
    assert not hasattr(LayerState(), "ple_tokens")


def test_histories_are_independent() -> None:
    state = TTState(num_layers=2, batch=2)
    state.histories[0].append(7)
    assert state.histories == [[7], []], "sequences must not share a list"


def test_position_property_reports_the_first_sequence() -> None:
    state = TTState(num_layers=1, batch=2)
    state.positions = [5, 9]
    assert state.position == 5


def test_layer_state_starts_empty() -> None:
    """Every buffer is allocated lazily, on first use, at its final size."""
    layer = LayerState()
    for field in ("conv", "recurrent", "keys", "values", "indexer_blocks", "indexer_ring", "ple_conv"):
        assert getattr(layer, field) is None, field
