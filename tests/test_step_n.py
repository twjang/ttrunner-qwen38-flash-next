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

from ttrunner_qwen38_flash_next.tt.model import TTModel  # noqa: E402


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


def test_attention_reads_per_row_with_a_tensor_position() -> None:
    """Row i reads with `cur_pos = start + i`, which is its causal window and
    already contains the rows before it -- every row is written first.

    Not the chunk path's masked `scaled_dot_product_attention`: that slices the
    cache to `start + k` rounded up to a tile, so its shapes grow with position
    and a captured trace would only be valid inside one tile."""
    src = inspect.getsource(TTModel._attention_step_n)
    assert "scaled_dot_product_attention_decode" in src
    assert "cur_pos_tensor=idxs[i]" in src
    assert "kpos <= qpos" not in src, "no growing mask"
    assert src.index("paged_update_cache") < src.index("for i in range(k):")


def test_step_n_refuses_while_it_lacks_the_sparse_selection() -> None:
    """Dense here while `step` runs sparse would make the two disagree beyond
    the budget, silently."""
    src = inspect.getsource(TTModel.step_n)
    assert "if self.use_indexer:" in src
    assert "does not carry the QSA selection yet" in src


def test_step_n_is_single_sequence_and_bounded() -> None:
    """One sequence, and k bounded by correctness rather than by the batch cliff.

    The bound used to be 64 -- "65 rows pad to 96 tiles and overflow L1" -- which
    is a real limit but not the binding one: `step_n` is wrong from k=33. See
    `test_step_n_refuses_k_past_one_tile_of_rows`.
    """
    src = inspect.getsource(TTModel.step_n)
    assert 'raise NotImplementedError("step_n advances one sequence at a time")' in src
    assert "else 32" in src


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


def test_the_capture_warms_the_single_token_step_too() -> None:
    """A caller that speculates still takes ordinary steps -- for a round with no
    draft, and to replay a rejected one. Those would otherwise allocate the whole
    48-layer step graph after the capture, which is the trace hazard: the engine
    hung on its first eager step and the boards needed `tt-smi -r`."""
    import inspect

    from ttrunner_qwen38_flash_next.tt.traced import TracedStepN

    src = inspect.getsource(TracedStepN.__init__)
    warm = src[: src.index("begin_trace_capture")]
    assert "model.step([warmup_token] * state.batch, state)" in warm
    assert "model.logits(" in warm


def test_step_n_refuses_k_past_one_tile_of_rows() -> None:
    """`step_n` is wrong from k=33, so the guard stops at 32, not the batch cliff.

    Measured by `scripts/dev/step_n_check.py`: exact at k=1, 2, 4, 8 and 16
    (0.00 % on the hidden, tokens matching, positions right) and 35.68 % out at
    k=33, 48 and 64 — the same figure at all three, so it is a structural break
    at one tile of rows rather than something that accumulates. The range had
    said 1..64 while nothing exercised past 8.

    Not the MoE: `moe_rows_check.py` puts `moe_block` at 64 rows within one row
    of its per-row answer, and that row is 5.8's routing tie.
    """
    import inspect

    from ttrunner_qwen38_flash_next.tt.model import TTModel

    src = inspect.getsource(TTModel.step_n)
    assert "else 32" in src, "the guard must stop at one tile of rows by default"
    assert "TWTEST_ALLOW_WIDE_STEP_N" in src, "with a documented escape for investigation"
