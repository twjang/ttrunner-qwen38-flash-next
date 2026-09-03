"""Chunked prefill: shape contracts it has to honour, and its correctness status.

Prefill is 11.3x faster than feeding a prompt through the decode path (5.77 s vs
65.15 s for 128 tokens), which is what makes a long context usable at all. It is
not enabled: it disagrees with the decode path from the very first chunk, so it
is fast and wrong, and `TTEngine` refuses `chunked_prefill`.

These lock the contracts that were actually broken, so the next attempt starts
from shapes that fit rather than from a kernel-level error.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("ttnn")

from twtest.tt.engine import TTEngine  # noqa: E402
from twtest.tt.model import TTModel  # noqa: E402


def test_chunk_paths_speak_the_ring_representation() -> None:
    """Decode keeps the conv windows as rings of columns; prefill must too.

    Neither decode mode lays a ring out oldest-at-index-0: trace-safe keeps the
    newest at index 0, and the rotating path leaves the oldest at
    `step % len(ring)`. A chunk that assumes either one permutes the taps
    silently, so both chunk paths go through the shared helpers, which honour
    the counter, and both leave the counter advanced.
    """
    for fn in (TTModel._causal_conv_chunk, TTModel._ple_chunk):
        src = inspect.getsource(fn)
        assert "_ring_oldest_first(" in src, f"{fn.__name__} must read the ring by its counter"
        assert "_ring_from_oldest_first(" in src, f"{fn.__name__} must write the ring by its counter"


def test_ring_helpers_round_trip_in_both_modes() -> None:
    """`_ring_from_oldest_first` is the inverse of `_ring_oldest_first`, which
    is what makes a chunk leave the state a decode step would have left."""
    model = TTModel.__new__(TTModel)
    cols = ["a", "b", "c", "d"]                     # oldest .. newest
    for trace_safe in (False, True):
        model.trace_safe_rings = trace_safe
        for step in range(9):
            ring = model._ring_from_oldest_first(cols, step)
            assert model._ring_oldest_first(ring, step) == cols, (trace_safe, step)


def _prefill_source() -> str:
    """`prefill` plus the FFN half it delegates to.

    The MoE half moved into `_prefill_ffn_half` so the single-chunk and grouped
    prefill paths share one body. These are invariants of the prefill path, so
    they are checked against all of it.
    """
    return inspect.getsource(TTModel.prefill) + inspect.getsource(TTModel._prefill_ffn_half)


def _deltanet_chunk_source() -> str:
    """The chunk path is split across three methods, so read all of them.

    `_linear_attention_chunk` was one function until the DeltaNet scan was
    batched over chunks (`_linear_attention_multi`). The invariants below are
    properties of the *path*, not of any one function, so they are checked
    against the whole of it.
    """
    return "".join(
        inspect.getsource(fn)
        for fn in (
            TTModel._deltanet_front,
            TTModel._deltanet_scan,
            TTModel._deltanet_back,
            TTModel._linear_attention_chunk,
            TTModel._linear_attention_multi,
        )
    )


def test_a_chunk_advances_the_ring_counters() -> None:
    assert "new_step = step + seq" in inspect.getsource(TTModel._causal_conv_chunk)
    assert "st.ple_step += seq" in inspect.getsource(TTModel._ple_chunk)
    # and the DeltaNet caller stores it back
    assert "st.conv_step = self._causal_conv_chunk(" in _deltanet_chunk_source().replace(
        "conv_out, st.conv, ", ""
    )


def test_chunked_deltanet_uses_local_head_counts() -> None:
    """The DeltaNet weights are head-sharded; global counts prepare one device."""
    src = _deltanet_chunk_source()
    assert "self.n_v_local" in src and "self.conv_dim_local" in src


def test_chunked_deltanet_never_leaves_the_device() -> None:
    """No host round trip in the chunk path -- for correctness, then for tracing.

    This began as a guard on a gather bug: the preparation ran on the host, and
    reading the sharded q/k/v with `from_dev` returned device 0's heads, so
    every device ran the recurrence on device 0's data. `prepare_device` removes
    the question -- each device prepares the heads it already holds, and there
    is no gather to get wrong.

    It also removes the reason chunked prefill could not be traced. A capture
    records device ops; host arithmetic between them is invisible, so a graph
    with `prepare()` in the middle replays as garbage. Keep this path free of
    `to_torch`/`from_torch` or that comes back.
    """
    src = _deltanet_chunk_source()
    # Positive anchor first: a test of "X is absent" passes on a gutted function,
    # so it has to prove it is reading the live path before the absence means
    # anything.
    assert "prepare_device(" in src, "the chunk must still prepare the op's inputs on device"
    for host_op in ("to_torch", "from_torch", "ShardTensorToMesh", "ConcatMeshToTensor"):
        assert host_op not in src, f"{host_op} puts the host back in the chunk path"


def test_prefill_selects_the_same_expert_weights_as_decode() -> None:
    """Naming the split halves while fused loads ~11 GB more and exhausts DRAM."""
    src = _prefill_source()
    assert "fuse_expert_gate_up" in src and "fused_gate_up" in src


def test_engine_allows_chunked_prefill_only_with_one_slot() -> None:
    """`TTModel.prefill` consumes a whole prompt before returning, which a
    shared lockstep batch cannot express -- so it is available at
    max_concurrency=1 and refused loudly above it, rather than silently
    ignored."""
    src = inspect.getsource(TTEngine.__init__)
    assert "chunked_prefill and max_concurrency != 1" in src
    assert "NotImplementedError" in src
    assert "self._chunked_prefill = bool(chunked_prefill)" in src


def test_prefill_leaves_the_last_prompt_token_to_the_decode_loop() -> None:
    """The step that consumes it is what produces the first output logits, and
    a recurrent state cannot be rewound to get it back. Prefill also has to
    stop on a tile boundary, because `fill_cache` asserts a tile-aligned
    index."""
    src = inspect.getsource(TTEngine._device_loop)
    block = src[src.index("if self._chunked_prefill:"):]
    assert "len(prompt) - 1 - seq.prompt_pos" in block
    assert "available % TILE" in block
    assert "seq.prompt_pos % TILE == 0" in block


def test_prefill_resumes_from_where_the_slot_already_is() -> None:
    """So it composes with prefix reuse instead of assuming position 0."""
    src = _prefill_source()
    assert "base = state.positions[0]" in src
    # the chunk's absolute start is base plus its offset within the prompt; the
    # loop variable is the group start since chunks run in groups
    assert "base + gstart" in src
    assert "state.positions = [base + gstart + len(gids)]" in src


def test_chunk_rings_are_written_in_place() -> None:
    """Rebinding a ring moves it, and a captured trace replays against the
    addresses it recorded -- so a prefill that rebound the rings would leave a
    traced decoder reading whatever used to be there."""
    src = inspect.getsource(TTModel._ring_from_oldest_first)
    assert "ttnn.copy(src, dst)" in src
    assert "into=state" in inspect.getsource(TTModel._causal_conv_chunk)
    assert "into=st.ple_conv" in inspect.getsource(TTModel._ple_chunk)


def test_chunked_deltanet_reduces_the_sharded_output() -> None:
    """`ssm_out` is row-sharded: without the all-reduce each device keeps a quarter.

    Found by per-layer bisection: the recurrent state (which never passes through
    `ssm_out`) matched decode while the layer output did not.
    """
    src = _deltanet_chunk_source()
    assert "self.all_reduce(" in src


def test_chunked_conv_taps_are_keyed_by_layer() -> None:
    """Keyed by channel count alone, every DeltaNet layer reused layer 0's taps."""
    src = inspect.getsource(TTModel._causal_conv_chunk)
    assert '("ssm_chunk", layer)' in src
    assert '("ssm_chunk", channels)' not in src



def _code_of(fn) -> str:
    """Source with comments stripped.

    These contracts assert on op names, and the code they guard carries a long
    comment naming the ops it deliberately no longer calls -- the tile-padding
    mask, the `repeat_interleave` -- because that history is why the current
    form is what it is. Matching on raw source would make the explanation fail
    the test that the explanation exists for.
    """
    lines = []
    for line in inspect.getsource(fn).split("\n"):
        stripped = line.split("#", 1)[0] if "#" in line else line
        lines.append(stripped)
    return "\n".join(lines)

def test_chunked_attention_builds_no_host_mask() -> None:
    """There must be no additive attention mask in the chunk path at all.

    There used to be, and it was wrong: the K/V slice and the mask were both
    TILE_LAYOUT, so a key length that was not a multiple of 32 got padded --
    with zeros, which an additive mask reads as "attend to me". The softmax
    spread over up to 31 all-zero keys and the block returned roughly v/32
    instead of v (`docs/iterations/013`). It was fixed by slicing to a whole
    tile; it is now *impossible*, because
    `chunked_scaled_dot_product_attention` is causal internally and there is no
    mask to pad.

    Keeping this as a test rather than deleting it: a future change that
    reintroduces an explicit mask here reintroduces the padding question with
    it, and the failure was silent the first time.
    """
    src = _code_of(TTModel._attention_chunk)
    # Positive anchor: absence proves nothing about a function that no longer
    # does the work.
    assert "chunked_scaled_dot_product_attention" in src, "still the causal chunked op"
    assert "attn_mask" not in src, "the chunk path must not build an attention mask"
    assert "torch.where" not in src and "torch.arange" not in src, (
        "no host-built mask or position vector in the chunk path"
    )
    assert "repeat_interleave" not in src, (
        "chunked SDPA reads n_kv directly; expanding KV heads is wasted work"
    )


def test_chunked_attention_takes_its_position_from_the_device() -> None:
    """The chunk start must be a device tensor, or the path cannot be captured.

    A trace records ops, not Python values: `fill_cache(update_idx=start)` and a
    host-built mask bake one chunk's position into the recorded program, so
    every replay rewrites the same slot. The paged ops take the position as
    data -- a page table for the write, `chunk_start_idx_tensor` for the read --
    which is what makes one capture serve every chunk (handoff 5.2).
    """
    src = _code_of(TTModel._attention_chunk)
    assert "chunk_start_idx_tensor" in src
    assert 'self._input(' in src, "the start must go through the bound-buffer path"
    assert "ttnn.fill_cache(" not in src, "fill_cache takes a Python int position"
    assert "paged_fill_cache" in src


def test_chunked_prefill_turns_the_trace_off() -> None:
    """Prefill runs eagerly and allocates gigabytes per call; a trace replays
    against the addresses it captured. After a few prefills the replay returned
    token 0 repeatedly -- eager prefill is correct, the combination is not."""
    src = inspect.getsource(TTEngine.__init__)
    assert "and not self._chunked_prefill" in src


def test_prefill_refuses_a_moe_chunk_past_the_cliff() -> None:
    """`moe_chunk` above 32 changes the answer, so it is refused, not documented.

    The fastest setting measured (64, 136.7 tok/s against 116.3 at 32) is on the
    wrong side of it: next-token top-1 on a 128-token chunk falls from 53.1 % to
    43.8 % at 64 and 21.9 % at 128, while 8, 16 and 32 agree on every token. A
    silent 10-point loss for a 1.2x speedup is not a trade a caller can make by
    accident.
    """
    import twtest.tt.model as model_mod

    assert model_mod._MAX_MOE_CHUNK == 32
    src = _prefill_source()
    assert "_MAX_MOE_CHUNK" in src, "the cap must be enforced, not just recorded"
    assert "moe_chunk: int = 32" in src.split("\n")[0] or "moe_chunk: int = 32" in src, (
        "the default must sit inside the verified range"
    )


def test_prefill_routes_in_groups_but_computes_experts_once() -> None:
    """The MoE's two halves take different row-group sizes, on purpose.

    Routing at a wider group changes the answer -- bf16 moves the router's
    probabilities ~0.5 %, reordering experts across the k-th boundary -- so it
    stays in `moe_chunk`-sized groups. The expert FFN is exactly per-token at
    any width once `sparse_program_config` gives it a single K block, so it runs
    once for the whole chunk: 612 fewer dispatches and 1070.5 -> 925.1 ms.

    Collapsing these back into one `moe_block` call per sub-chunk is the
    regression this guards.
    """
    src = _prefill_source()
    assert "moe.route(" in src, "routing must stay per sub-chunk"
    assert "moe.apply_experts(" in src, "the expert FFN must run once for the chunk"
    assert "moe.moe_block(" not in src, (
        "moe_block fuses routing and expert compute at one group size"
    )
