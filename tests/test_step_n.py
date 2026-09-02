"""`step_n` advances one sequence by k tokens in a single step.

A step is flat in batch up to 64 rows, so the k tokens ride the batch axis for
everything that is per-token. Only two things cannot: the DeltaNet convolution,
whose window is the three rows before it, and the recurrence, whose state is the
previous row's. Those are unrolled, and everything else is untouched.

Measured on device (`scripts/dev/step_n_check.py`, `step_n_bench.py`): the hidden
matches the sequential path exactly at every one of the k positions, and k=4
costs 864 ms against 2071 for four steps.

Host-side: these lock the structure, not the numbers.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("ttnn")

from twtest.tt.model import TTModel  # noqa: E402


def test_only_the_convolution_and_the_recurrence_are_unrolled() -> None:
    """Everything else is per-token and must stay batched, or the win is gone."""
    src = inspect.getsource(TTModel._linear_attention_step_n)
    # the two sequential pieces, each in its own loop over the k rows
    assert src.count("for i in range(k):") == 2
    # the projections, the gate and ssm_out are outside any loop
    body = src[: src.index("qkv_col = ")]
    assert 'self.w.blk(layer, "attn_qkv.weight")' in body
    assert 'self.w.blk(layer, "attn_gate.weight")' in body
    assert "for i in range" not in body


def test_the_recurrence_carries_one_state_across_the_rows() -> None:
    """k rows of one sequence share a state; k *sequences* would not."""
    src = inspect.getsource(TTModel._linear_attention_step_n)
    assert "(n_v, 1, hd, hd)" in src, "the state is single-sequence, not k * n_v"
    assert src.count("st.recurrent,") == 1, "one shared state, threaded through the loop"


def test_attention_writes_the_cache_one_row_at_a_time() -> None:
    """`fill_cache` asserts a tile-aligned index, so the chunk path cannot start
    at an arbitrary position; `paged_update_cache` takes the index as a tensor,
    which also keeps the step traceable."""
    src = inspect.getsource(TTModel._attention_step_n)
    assert "paged_update_cache" in src
    assert "ttnn.fill_cache(" not in src, "only the docstring may mention it"
    assert "for i, pos in enumerate(positions):" in src


def test_attention_masks_causally_among_the_k_rows() -> None:
    """Row i may see rows 0..i and everything before the chunk, nothing after."""
    src = inspect.getsource(TTModel._attention_step_n)
    assert "kpos <= qpos" in src
    assert "ttnn.TILE_SIZE" in src, "the K/V slice has to be a whole tile"


def test_step_n_is_single_sequence_and_bounded_by_the_batch_cliff() -> None:
    src = inspect.getsource(TTModel.step_n)
    assert 'raise NotImplementedError("step_n advances one sequence at a time")' in src
    assert "0 < k <= 64" in src, "65 rows pad to 96 tiles and overflow L1"


def test_step_n_grows_the_history_per_row() -> None:
    """Row i's PLE n-gram hash reads the history up to and including token i."""
    src = inspect.getsource(TTModel.step_n)
    assert "histories.append(list(base))" in src
    assert "state.histories[0] = base" in src


def test_step_n_advances_the_position_by_k() -> None:
    assert "state.positions = [start + k]" in inspect.getsource(TTModel.step_n)


def test_the_moe_block_is_shared_with_the_single_token_step() -> None:
    """It is per-token, so both paths must run the same code -- an MoE that
    drifted between them is exactly the sort of thing that stays hidden."""
    assert "self._moe_block(mixed, layer)" in inspect.getsource(TTModel._layer)
    assert "self._moe_block(mixed, layer)" in inspect.getsource(TTModel.step_n)
