"""Provider abstraction: one interface over every local LLM backend."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from localcode.config.schema import ProviderConfig
from localcode.errors import ProviderError
from localcode.logging import get_logger
from localcode.providers import toolparse

log = get_logger("provider")

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ImageContent:
    """Base64-encoded image attached to a user message."""

    data: str       # base64 bytes, no data-URI prefix
    mime_type: str  # "image/png", "image/jpeg", "image/gif", "image/webp"


@dataclass
class Message:
    role: Role
    content: str = ""
    images: list[ImageContent] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool" messages
    name: str | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StreamEvent:
    kind: Literal["text", "tool_call", "usage"]
    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None


class Provider(ABC):
    """A chat backend. Subclasses implement the wire protocol; this base
    class implements tool-mode selection and the JSON tool-call fallback."""

    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg
        self._native_tools_broken = False

    @property
    def model(self) -> str:
        return self.cfg.model

    @abstractmethod
    def _chat_native(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion using the backend's own tool-calling API.

        Implementations must raise ProviderError(retryable=...) on failure.
        """

    @abstractmethod
    async def list_models(self) -> list[str]: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion, handling tool_mode and native-API fallback."""
        mode = self.cfg.tool_mode
        use_json = mode == "json" or (mode == "auto" and self._native_tools_broken)
        if not tools:
            use_json = False  # nothing to translate

        if not use_json:
            try:
                got_output = False
                async for event in self._chat_native(messages, tools):
                    got_output = True
                    yield event
                return
            except ProviderError as e:
                if mode != "auto" or got_output or not tools:
                    raise
                log.warning(
                    "native tool calling failed, falling back to json mode",
                    extra={"data": {"error": str(e)}},
                )
                self._native_tools_broken = True

        # JSON fallback: describe tools in the system prompt and parse
        # <tool_call> blocks out of the streamed text.
        assert tools is not None
        prompted = toolparse.inject_tools_prompt(messages, tools)
        extractor = toolparse.StreamToolExtractor()
        usage: Usage | None = None
        async for event in self._chat_native(prompted, None):
            if event.kind == "text":
                visible = extractor.feed(event.text)
                if visible:
                    yield StreamEvent(kind="text", text=visible)
            elif event.kind == "usage":
                usage = event.usage
        tail, calls = extractor.finalize()
        if tail:
            yield StreamEvent(kind="text", text=tail)
        for call in calls:
            yield StreamEvent(kind="tool_call", tool_call=call)
        if usage is not None:
            yield StreamEvent(kind="usage", usage=usage)

    async def complete(self, messages: list[Message]) -> str:
        """Non-streaming convenience used for summarization etc."""
        parts: list[str] = []
        async for event in self._chat_native(messages, None):
            if event.kind == "text":
                parts.append(event.text)
        return "".join(parts)


async def with_retries[T](
    fn: Any, *, max_retries: int = 3, base_delay: float = 1.0
) -> T:
    """Run an async callable, retrying retryable provider errors with backoff."""
    attempt = 0
    while True:
        try:
            return await fn()  # type: ignore[no-any-return]
        except ProviderError as e:
            attempt += 1
            if not e.retryable or attempt > max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            log.info(f"retrying provider call in {delay:.1f}s ({attempt}/{max_retries}): {e}")
            await asyncio.sleep(delay)


def parse_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Tool-call arguments arrive as JSON text or dicts depending on backend."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    raw = raw.strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_raw": raw}


def create_provider(cfg: ProviderConfig) -> Provider:
    """Factory: instantiate the configured provider type."""
    from localcode.providers.ollama import OllamaProvider
    from localcode.providers.openai_compat import OpenAICompatProvider

    if cfg.type == "ollama":
        return OllamaProvider(cfg)
    if cfg.type == "openai":
        return OpenAICompatProvider(cfg)
    raise ProviderError(f"Unknown provider type: {cfg.type}")
