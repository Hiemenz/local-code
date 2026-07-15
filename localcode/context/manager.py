"""Context management: token budgeting and conversation compaction.

Token counts use a chars/4 heuristic — cheap, tokenizer-free, and accurate
enough for budget decisions across model families.

To maximize server-side prompt/KV-cache hits, compaction rewrites history
as [system, summary, recent...] and otherwise never reorders messages, so
the prompt prefix stays stable between turns.
"""

from __future__ import annotations

import json

from localcode.config.schema import ContextConfig
from localcode.logging import get_logger
from localcode.providers.base import Message, Provider

log = get_logger("context")

SUMMARY_PROMPT = (
    "Summarize the conversation so far for a coding assistant that is about to "
    "continue the work. Preserve: the user's goals and constraints, decisions "
    "made, files created/modified (with paths), key code details, commands run "
    "and their outcomes, and any unresolved problems or next steps. Be dense "
    "and factual. Output only the summary."
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def message_tokens(msg: Message) -> int:
    total = estimate_tokens(msg.content) + 8
    for call in msg.tool_calls:
        total += estimate_tokens(call.name + json.dumps(call.arguments, default=str)) + 8
    return total


class ContextManager:
    def __init__(self, cfg: ContextConfig, context_length: int) -> None:
        self.cfg = cfg
        self.context_length = context_length

    def total_tokens(self, messages: list[Message]) -> int:
        return sum(message_tokens(m) for m in messages)

    def needs_compaction(self, messages: list[Message]) -> bool:
        return self.total_tokens(messages) > self.context_length * self.cfg.compact_threshold

    def _split_point(self, messages: list[Message]) -> int:
        """Index where the 'recent' window begins. Never splits a turn:
        the window must start at a user message (not a tool result)."""
        non_system = [i for i, m in enumerate(messages) if m.role != "system"]
        if len(non_system) <= self.cfg.keep_recent_messages:
            return -1
        target = non_system[-self.cfg.keep_recent_messages]
        for i in range(target, len(messages)):
            if messages[i].role == "user" and messages[i].tool_call_id is None:
                return i
        return -1

    async def compact(self, messages: list[Message], provider: Provider) -> list[Message]:
        """Summarize everything but the system prompt and the most recent turn(s)."""
        split = self._split_point(messages)
        if split <= 0:
            return messages
        head = [m for m in messages[:split] if m.role == "system"]
        middle = [m for m in messages[:split] if m.role != "system"]
        recent = messages[split:]
        if not middle:
            return messages

        transcript = _render_transcript(middle)
        summary = await provider.complete(
            [
                Message(role="system", content=SUMMARY_PROMPT),
                Message(role="user", content=transcript[-60_000:]),
            ]
        )
        summary_msg = Message(
            role="user",
            content=f"[Conversation summary — earlier context was compacted]\n{summary.strip()}",
        )
        compacted = [*head, summary_msg, *recent]
        log.info(
            "compacted context",
            extra={"data": {
                "before_tokens": self.total_tokens(messages),
                "after_tokens": self.total_tokens(compacted),
            }},
        )
        return compacted


def _render_transcript(messages: list[Message]) -> str:
    parts: list[str] = []
    for m in messages:
        if m.role == "tool":
            parts.append(f"[tool result: {m.name}]\n{m.content[:2000]}")
        elif m.tool_calls:
            calls = ", ".join(
                f"{c.name}({json.dumps(c.arguments, default=str)[:200]})" for c in m.tool_calls
            )
            parts.append(f"assistant: {m.content}\n[called: {calls}]")
        else:
            parts.append(f"{m.role}: {m.content}")
    return "\n\n".join(parts)
