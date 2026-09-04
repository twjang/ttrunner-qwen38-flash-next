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


# --------------------------------------------------------------------------
# tool calls
# --------------------------------------------------------------------------
import json  # noqa: E402

from ttrunner_qwen38_flash_next.server.api import (  # noqa: E402
    _ToolCallGate,
    parse_tool_calls,
)

CALL = (
    "<tool_call>\n<function=read>\n<parameter=file_path>\n/tmp/note.txt\n"
    "</parameter>\n</function>\n</tool_call>"
)


def test_a_tool_call_is_parsed_out_of_the_xml() -> None:
    """The checkpoint answers in its template's XML, not OpenAI tool-call JSON.

    Without this an agent harness sees the call as literal text and reads it as
    the model declining to use its tools.
    """
    content, calls = parse_tool_calls(CALL)
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"file_path": "/tmp/note.txt"}
    assert calls[0]["type"] == "function" and calls[0]["id"]


def test_prose_before_a_call_is_kept() -> None:
    """The template allows reasoning before a call, and it is part of the answer."""
    content, calls = parse_tool_calls("Let me look.\n" + CALL)
    assert content == "Let me look."
    assert len(calls) == 1


def test_parameters_recover_their_types() -> None:
    """XML has no types; a tool wanting `limit: 12` must not get "12"."""
    text = (
        "<tool_call>\n<function=grep>\n<parameter=pattern>\nhello world\n</parameter>\n"
        "<parameter=limit>\n12\n</parameter>\n<parameter=recursive>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    args = json.loads(parse_tool_calls(text)[1][0]["function"]["arguments"])
    assert args == {"pattern": "hello world", "limit": 12, "recursive": True}


def test_plain_text_is_untouched() -> None:
    assert parse_tool_calls("just an answer") == ("just an answer", [])


def test_the_gate_streams_prose_when_no_call_appears() -> None:
    gate = _ToolCallGate(True)
    streamed = "".join(gate.feed(c) for c in "hello there")
    trailing, calls = gate.flush()
    assert streamed + trailing == "hello there"
    assert calls == []


def test_the_gate_withholds_a_call_split_across_deltas() -> None:
    """The opening tag can straddle deltas, so the tail is held back."""
    gate = _ToolCallGate(True)
    streamed = "".join(gate.feed(d) for d in ("Look: ", "<tool", "_call>\n<function=read>\n"))
    streamed += gate.feed("<parameter=file_path>\n/tmp/note.txt\n</parameter>\n")
    streamed += gate.feed("</function>\n</tool_call>")
    trailing, calls = gate.flush()
    assert "<tool_call>" not in streamed + trailing
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read"


def test_the_gate_is_a_passthrough_when_no_tools_were_offered() -> None:
    gate = _ToolCallGate(False)
    assert gate.feed("<tool_call> literal") == "<tool_call> literal"


def test_tool_call_arguments_are_parsed_for_the_template() -> None:
    """OpenAI sends arguments as a JSON string; this template demands an object.

    Unfixed, the first turn of an agent loop works and the second returns 400,
    which a harness reports as an empty answer rather than an error.
    """
    from ttrunner_qwen38_flash_next.server.api import normalise_tool_calls

    msgs = normalise_tool_calls(
        [{"role": "assistant", "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "read", "arguments": '{"file_path": "/tmp/x"}'}}]}]
    )
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == {"file_path": "/tmp/x"}


def test_unparsable_arguments_are_left_for_the_template_to_reject() -> None:
    from ttrunner_qwen38_flash_next.server.api import normalise_tool_calls

    msgs = normalise_tool_calls(
        [{"role": "assistant", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "x", "arguments": "not json"}}]}]
    )
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "not json"
