"""Tool calling for models without a native tool API.

Tools are described in the system prompt; the model emits
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks
(the format Qwen models were trained on). ``StreamToolExtractor``
strips those blocks out of the token stream so the user never sees
raw JSON, even when a block spans many chunks.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from localcode.providers.base import Message, ToolCall, ToolSpec

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"

PROMPT_TEMPLATE = """

# Tools

You may call one or more tools to assist with the user request.
Available tools, as JSON schemas:

{tool_schemas}

To call a tool, emit exactly this format (one block per call), then stop:
<tool_call>
{{"name": "<tool-name>", "arguments": {{<args-json>}}}}
</tool_call>

Rules:
- Arguments must be valid JSON matching the tool's schema.
- Emit the block only when you actually want the tool executed.
- After tool results arrive, continue the task or answer the user."""


def render_tools_prompt(tools: list[ToolSpec]) -> str:
    schemas = "\n".join(
        json.dumps(
            {"name": t.name, "description": t.description, "parameters": t.parameters},
            ensure_ascii=False,
        )
        for t in tools
    )
    return PROMPT_TEMPLATE.format(tool_schemas=schemas)


def inject_tools_prompt(messages: list[Message], tools: list[ToolSpec]) -> list[Message]:
    """Return a copy of messages with tool instructions appended to the system
    message (or prepended as one), and tool-role messages converted to plain
    user messages that any model understands."""
    from localcode.providers.base import Message

    suffix = render_tools_prompt(tools)
    out: list[Message] = []
    have_system = False
    for msg in messages:
        if msg.role == "system" and not have_system:
            have_system = True
            out.append(Message(role="system", content=msg.content + suffix))
        elif msg.role == "tool":
            out.append(
                Message(
                    role="user",
                    content=f"<tool_result name={msg.name or ''!r}>\n{msg.content}\n</tool_result>",
                )
            )
        elif msg.role == "assistant" and msg.tool_calls:
            blocks = "".join(
                f"\n{OPEN_TAG}\n"
                + json.dumps({"name": c.name, "arguments": c.arguments}, ensure_ascii=False)
                + f"\n{CLOSE_TAG}"
                for c in msg.tool_calls
            )
            out.append(Message(role="assistant", content=(msg.content or "") + blocks))
        else:
            out.append(msg)
    if not have_system:
        out.insert(0, Message(role="system", content=suffix.strip()))
    return out


def _parse_call(body: str) -> ToolCall | None:
    from localcode.providers.base import ToolCall, parse_arguments

    try:
        data: Any = json.loads(body.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "name" not in data:
        return None
    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:12]}",
        name=str(data["name"]),
        arguments=parse_arguments(data.get("arguments")),
    )


def _possible_tag_prefix(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    max_len = min(len(text), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


class StreamToolExtractor:
    """Incrementally split streamed text into visible text and tool calls."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_block = False
        self._emitted = False
        self.calls: list[ToolCall] = []

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        # A response that starts with a bare JSON object is likely an untagged
        # tool call (some models drop the tags); hold it back until finalize.
        if not self._emitted and not self._in_block and self._buffer.lstrip().startswith("{"):
            return ""
        visible: list[str] = []
        while True:
            if self._in_block:
                end = self._buffer.find(CLOSE_TAG)
                if end < 0:
                    break
                call = _parse_call(self._buffer[:end])
                if call is not None:
                    self.calls.append(call)
                self._buffer = self._buffer[end + len(CLOSE_TAG):]
                self._in_block = False
            else:
                start = self._buffer.find(OPEN_TAG)
                if start >= 0:
                    visible.append(self._buffer[:start])
                    self._buffer = self._buffer[start + len(OPEN_TAG):]
                    self._in_block = True
                    continue
                hold = _possible_tag_prefix(self._buffer, OPEN_TAG)
                emit_upto = len(self._buffer) - hold
                visible.append(self._buffer[:emit_upto])
                self._buffer = self._buffer[emit_upto:]
                break
        out = "".join(visible)
        if out.strip():
            self._emitted = True
        return out

    def finalize(self) -> tuple[str, list[ToolCall]]:
        """Flush remaining text; also accept a bare JSON object as a call."""
        tail = self._buffer
        self._buffer = ""
        if self._in_block:
            # Unterminated block: try to parse what we have.
            call = _parse_call(tail)
            if call is not None:
                self.calls.append(call)
                tail = ""
            self._in_block = False
        elif not self.calls and tail.strip().startswith("{") and '"name"' in tail:
            call = _parse_call(tail)
            if call is not None:
                self.calls.append(call)
                tail = ""
        return tail, self.calls
