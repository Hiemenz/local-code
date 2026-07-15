import json

from localcode.providers.base import Message, ToolSpec
from localcode.providers.toolparse import (
    StreamToolExtractor,
    inject_tools_prompt,
    render_tools_prompt,
)

SPEC = ToolSpec(name="read_file", description="Read a file", parameters={"type": "object"})


def test_extractor_plain_text() -> None:
    ex = StreamToolExtractor()
    out = ex.feed("hello ") + ex.feed("world")
    tail, calls = ex.finalize()
    assert out + tail == "hello world"
    assert calls == []


def test_extractor_single_call() -> None:
    ex = StreamToolExtractor()
    payload = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    visible = ex.feed("Let me look. " + payload)
    tail, calls = ex.finalize()
    assert visible.strip() == "Let me look."
    assert tail == ""
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}


def test_extractor_split_across_chunks() -> None:
    ex = StreamToolExtractor()
    payload = 'before <tool_call>{"name": "grep", "arguments": {"pattern": "x"}}</tool_call> after'
    visible = "".join(ex.feed(payload[i : i + 3]) for i in range(0, len(payload), 3))
    tail, calls = ex.finalize()
    assert (visible + tail).strip() == "before  after".strip()
    assert len(calls) == 1
    assert calls[0].name == "grep"


def test_extractor_holds_back_partial_tag_then_releases() -> None:
    ex = StreamToolExtractor()
    first = ex.feed("value <tool")
    assert "<tool" not in first  # held back as a potential tag
    second = ex.feed("box>")  # turns out not to be a tool_call tag
    tail, calls = ex.finalize()
    assert first + second + tail == "value <toolbox>"
    assert calls == []


def test_extractor_multiple_calls() -> None:
    ex = StreamToolExtractor()
    ex.feed(
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    _, calls = ex.finalize()
    assert [c.name for c in calls] == ["a", "b"]


def test_finalize_bare_json_fallback() -> None:
    ex = StreamToolExtractor()
    ex.feed(json.dumps({"name": "bash", "arguments": {"command": "ls"}}))
    tail, calls = ex.finalize()
    assert tail == ""
    assert calls[0].name == "bash"


def test_finalize_unterminated_block() -> None:
    ex = StreamToolExtractor()
    ex.feed('<tool_call>{"name": "bash", "arguments": {"command": "ls"}}')
    tail, calls = ex.finalize()
    assert calls and calls[0].name == "bash"
    assert tail == ""


def test_inject_tools_prompt_converts_history() -> None:
    from localcode.providers.base import ToolCall

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "a"})],
        ),
        Message(role="tool", content="data", tool_call_id="1", name="read_file"),
    ]
    out = inject_tools_prompt(messages, [SPEC])
    assert "read_file" in out[0].content  # schema injected into system
    assert out[2].role == "assistant" and "<tool_call>" in out[2].content
    assert out[3].role == "user" and "tool_result" in out[3].content


def test_render_tools_prompt_lists_all() -> None:
    text = render_tools_prompt([SPEC, ToolSpec(name="bash", description="d", parameters={})])
    assert "read_file" in text and "bash" in text
