"""The chat endpoint must not hand a reasoning model's scratchpad to the caller.

This checkpoint's template leaves the assistant turn open inside `<think>`, so
generation begins as reasoning and only becomes the answer after `</think>`.
Before this was handled, "Reply with exactly: OK" came back as
'We need to respond to user: ... Ensure no extra.\\n' -- the thinking, with the
answer never reached inside the token budget.
"""
from ttrunner_qwen38_flash_next.server.api import (
    _ThinkSplitter,
    _thinking_is_open,
    split_thinking,
)


def test_open_block_is_read_off_the_rendered_prompt() -> None:
    """The template decides, so the prompt is what gets asked."""
    assert _thinking_is_open("<|im_start|>assistant\n<think>\n")
    # enable_thinking: false closes the block in the prompt itself
    assert not _thinking_is_open("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not _thinking_is_open("plain completion prompt")


def test_reasoning_is_split_from_the_answer() -> None:
    reasoning, content = split_thinking("thinking out loud\n</think>\n\nOK", True)
    assert reasoning == "thinking out loud\n"
    assert content == "OK"


def test_a_closed_block_is_all_answer() -> None:
    """With enable_thinking false the model never emits the tag."""
    assert split_thinking("OK", False) == ("", "OK")


def test_an_unfinished_thought_is_reasoning_not_answer() -> None:
    """Budget ran out mid-thought: content is empty, and that is the truth.

    Returning the scratchpad as the answer is the bug this file exists for.
    """
    reasoning, content = split_thinking("still thinking and never got to", True)
    assert content == ""
    assert reasoning == "still thinking and never got to"


def test_the_closing_tag_may_straddle_two_deltas() -> None:
    """Tokens do not respect tag boundaries, which is why the block is buffered."""
    sp = _ThinkSplitter(True)
    out = [sp.feed(d) for d in ("why", " not", "</th", "ink>", "\n\nAnswer")]
    reasoning = "".join(r for r, _ in out) + sp.flush()[0]
    content = "".join(c for _, c in out) + sp.flush()[1]
    assert reasoning == "why not"
    assert content == "Answer"


def test_answer_streams_once_the_block_closes() -> None:
    """After the tag, deltas pass straight through -- no buffering of the answer."""
    sp = _ThinkSplitter(True)
    sp.feed("thought</think>\n\n")
    assert sp.feed("one") == ("", "one")
    assert sp.feed(" two") == ("", " two")
