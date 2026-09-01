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
    """Decode keeps the conv windows as lists of columns; prefill must too."""
    for fn in (TTModel._causal_conv_chunk, TTModel._ple_chunk):
        src = inspect.getsource(fn)
        assert "trace_safe_rings" in src, (
            f"{fn.__name__} must pick the ring order the following decode mode "
            "reads: trace-safe reads newest-first, the rotating path oldest-first"
        )


def test_chunked_deltanet_uses_local_head_counts() -> None:
    """The DeltaNet weights are head-sharded; global counts prepare one device."""
    src = inspect.getsource(TTModel._linear_attention_chunk)
    assert "self.n_v_local" in src and "self.conv_dim_local" in src
    assert "ShardTensorToMesh" in src, (
        "each device's own heads must be prepared and sharded back; from_dev "
        "returns device 0's copy, which for a head-sharded tensor is its heads only"
    )


def test_prefill_selects_the_same_expert_weights_as_decode() -> None:
    """Naming the split halves while fused loads ~11 GB more and exhausts DRAM."""
    src = inspect.getsource(TTModel.prefill)
    assert "fuse_expert_gate_up" in src and "fused_gate_up" in src


def test_engine_refuses_chunked_prefill_while_it_is_wrong() -> None:
    src = inspect.getsource(TTEngine.__init__)
    assert "chunked_prefill" in src and "NotImplementedError" in src
