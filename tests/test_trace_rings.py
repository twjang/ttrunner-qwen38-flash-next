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

from ttrunner_qwen38_flash_next.tt.model import TTModel  # noqa: E402
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder  # noqa: E402


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


def test_engine_enables_trace_for_the_single_user_path() -> None:
    """Trace is the whole latency story at batch 1: 516 ms -> 255 ms.

    It was off by default while a captured decoder was correct through TTModel
    and corrupt through TTEngine. The cause was allocation ordering, not the
    graph: `output.weight` loads lazily and greedy_tokens/logits allocate their
    own intermediates, and in the engine all of that first happened *after*
    `begin_trace_capture`, landing on memory the recorded graph depends on.
    Every standalone harness happened to preload the weights and call the LM head
    before capturing, which is why none of them could reproduce it.
    """
    import inspect

    from ttrunner_qwen38_flash_next.tt.engine import TTEngine
    from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder

    sig = inspect.signature(TTEngine.__init__)
    assert sig.parameters["use_trace"].default is True

    src = inspect.getsource(TracedDecoder.__init__)
    assert "model.logits(" in src and "greedy_tokens(" in src, (
        "TracedDecoder must exercise the LM head before capturing; allocating it "
        "afterwards corrupts the replay"
    )
    assert src.index("model.logits(") < src.index("begin_trace_capture"), (
        "the LM head warmup must come before the capture region"
    )
