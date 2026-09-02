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


def test_a_chunk_advances_the_ring_counters() -> None:
    assert "new_step = step + seq" in inspect.getsource(TTModel._causal_conv_chunk)
    assert "st.ple_step += seq" in inspect.getsource(TTModel._ple_chunk)
    # and the DeltaNet caller stores it back
    assert "st.conv_step = self._causal_conv_chunk(" in inspect.getsource(
        TTModel._linear_attention_chunk
    ).replace("conv_out, st.conv, ", "")


def test_chunked_deltanet_uses_local_head_counts() -> None:
    """The DeltaNet weights are head-sharded; global counts prepare one device."""
    src = inspect.getsource(TTModel._linear_attention_chunk)
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
    src = inspect.getsource(TTModel._linear_attention_chunk)
    for host_op in ("to_torch", "from_torch", "ShardTensorToMesh", "ConcatMeshToTensor"):
        assert host_op not in src, f"{host_op} puts the host back in the chunk path"


def test_prefill_selects_the_same_expert_weights_as_decode() -> None:
    """Naming the split halves while fused loads ~11 GB more and exhausts DRAM."""
    src = inspect.getsource(TTModel.prefill)
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
    src = inspect.getsource(TTModel.prefill)
    assert "base = state.positions[0]" in src
    assert "base + begin" in src


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
    src = inspect.getsource(TTModel._linear_attention_chunk)
    assert "self.all_reduce(" in src


def test_chunked_conv_taps_are_keyed_by_layer() -> None:
    """Keyed by channel count alone, every DeltaNet layer reused layer 0's taps."""
    src = inspect.getsource(TTModel._causal_conv_chunk)
    assert '("ssm_chunk", layer)' in src
    assert '("ssm_chunk", channels)' not in src


def test_chunked_attention_masks_the_tile_padding() -> None:
    """The K/V slice and the additive mask are TILE_LAYOUT, so a key length that
    is not a multiple of 32 is padded -- with zeros, which an additive mask reads
    as "attend to me". Slicing to a whole tile and running the causal condition
    over that width is what masks the pad; a slice to `total` does not."""
    src = inspect.getsource(TTModel._attention_chunk)
    assert "ttnn.TILE_SIZE" in src
    assert "(1, n_kv, kv_len, hd)" in src
    assert "torch.arange(kv_len)" in src
    assert "(1, n_kv, total, hd)" not in src


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
    src = inspect.getsource(TTModel.prefill)
    assert "_MAX_MOE_CHUNK" in src, "the cap must be enforced, not just recorded"
    assert "moe_chunk: int = 32" in src.split("\n")[0] or "moe_chunk: int = 32" in src, (
        "the default must sit inside the verified range"
    )
