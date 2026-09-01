"""Trace capture records device ops only -- host-side rotation is invisible to it.

The conv windows advance in the eager path by rebinding a Python list slot
(`ring[step % depth] = col`). That is free and correct when every step runs
eagerly, and silently wrong under trace: capture records no rotation, so a
replayed step reads a convolution history frozen at capture time. Measured
directly -- traced output diverged from eager at the *first* generated token,
while the trace benchmarks, which timed the step without checking what it
computed, reported a 1.5x speedup on the wrong answer.

`TTModel.trace_safe_rings` switches the rings to fixed read indices plus a
device-copy shift, which capture does record.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("ttnn")

from twtest.tt.model import TTModel  # noqa: E402
from twtest.tt.traced import TracedDecoder  # noqa: E402


def test_traced_decoder_enables_trace_safe_rings_before_capture() -> None:
    src = inspect.getsource(TracedDecoder.__init__)
    assert "trace_safe_rings = True" in src, (
        "TracedDecoder must switch the rings to the device-copy shift; without it "
        "a replayed step reads a convolution history frozen at capture time"
    )
    # it has to happen before the warmup step allocates and advances the rings
    assert src.index("trace_safe_rings = True") < src.index("model.step("), (
        "trace_safe_rings must be set before the warmup step, or capture and "
        "replay disagree about how the rings advance"
    )


@pytest.mark.parametrize("fn", [TTModel._causal_conv_step, TTModel._ple_step])
def test_both_rings_have_a_trace_safe_path(fn) -> None:
    src = inspect.getsource(fn)
    assert "trace_safe_rings" in src, f"{fn.__name__} has no trace-safe rotation"
    assert "ttnn.copy" in src, (
        f"{fn.__name__}'s trace-safe path must shift with device copies; a Python "
        "rebinding is not recorded by trace capture"
    )


def test_reset_handles_ring_lists() -> None:
    """The rings became lists; reset() zeroed them as if they were tensors."""
    src = inspect.getsource(TracedDecoder.reset)
    assert "isinstance(buf, list)" in src, (
        "TracedDecoder.reset must clear each column of a ring; conv/ple_conv are "
        "lists of tensors, and calling .shape on the list raises"
    )


def test_engine_does_not_enable_trace_by_default() -> None:
    """Trace is correct through TTModel and wrong through TTEngine.

    Standalone, a captured decoder matches eager token-for-token (2.05x at batch
    1, 1.35x at 32). Driven through TTEngine the same decoder emits garbage --
    ' Paris.' becomes '!!!!'. Every candidate was excluded by a standalone
    reproduction that passed: reset_slot between replays, repeated post-capture
    allocation, max_seq_len, trace region size, distinct per-slot tokens, the
    filler-token pattern, the live slot's index, and the worker-thread boundary.

    Until a failing reproduction exists outside the engine, the default stays off:
    a wrong answer served fast is worse than a right one served slower.
    """
    import inspect

    from twtest.tt.engine import TTEngine

    sig = inspect.signature(TTEngine.__init__)
    assert sig.parameters["use_trace"].default is False, (
        "TTEngine must not enable trace by default -- the traced engine path "
        "produces corrupted output for a cause that is not yet isolated"
    )
