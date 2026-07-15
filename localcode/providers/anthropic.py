"""Anthropic Messages API provider.

Targets api.anthropic.com (or a compatible proxy) using the streaming Messages
API with native tool use.  No third-party SDK — just httpx + the public REST
spec so the dependency footprint stays minimal.

Wire-format differences from OpenAI that this module handles:
  * The system prompt lives in a top-level "system" field, not in messages.
  * Tool results are "tool_result" content blocks inside *user* messages;
    consecutive tool-result messages are batched into a single user turn.
  * Tool specs use "input_schema" (JSON Schema) instead of "parameters".
  * Streaming uses typed SSE events (message_start / content_block_* /
    message_delta) rather than delta-patched choice objects.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from localcode.config.schema import ProviderConfig
from localcode.errors import ProviderError
from localcode.providers.base import (
    Message,
    Provider,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
    parse_arguments,
)

_ANTHROPIC_DEFAULT = "https://api.anthropic.com"
_OLLAMA_DEFAULT = "http://localhost:11434"
_API_VERSION = "2023-06-01"


def _wire_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Convert internal messages to Anthropic wire format.

    Returns (system_text, wire_messages).
    """
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []

    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
            continue

        if m.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id or "",
                "content": m.content,
            }
            # Consecutive tool results must be batched into one user message.
            if (
                wire
                and wire[-1]["role"] == "user"
                and isinstance(wire[-1]["content"], list)
                and wire[-1]["content"]
                and wire[-1]["content"][0].get("type") == "tool_result"
            ):
                wire[-1]["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block]})
            continue

        if m.role == "user":
            parts: list[dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.mime_type,
                        "data": img.data,
                    },
                }
                for img in m.images
            ]
            if m.content:
                parts.append({"type": "text", "text": m.content})
            wire.append({"role": "user", "content": parts or [{"type": "text", "text": ""}]})
            continue

        if m.role == "assistant":
            parts = []
            if m.content:
                parts.append({"type": "text", "text": m.content})
            for c in m.tool_calls:
                parts.append({
                    "type": "tool_use",
                    "id": c.id,
                    "name": c.name,
                    "input": c.arguments,
                })
            wire.append({"role": "assistant", "content": parts or [{"type": "text", "text": ""}]})

    return "\n\n".join(system_parts), wire


class AnthropicProvider(Provider):
    """Anthropic Messages API (claude-* models) with streaming and native tool use."""

    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        api_key = cfg.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        # Fall back to the Anthropic API when the user left base_url at the
        # Ollama default (most likely because they only set type = "anthropic").
        base = (
            _ANTHROPIC_DEFAULT
            if cfg.base_url == _OLLAMA_DEFAULT
            else cfg.base_url.rstrip("/")
        )
        self._client = httpx.AsyncClient(
            base_url=base,
            headers={
                "x-api-key": api_key,
                "anthropic-version": _API_VERSION,
                "content-type": "application/json",
                **cfg.extra_headers,
            },
            timeout=httpx.Timeout(cfg.timeout_seconds),
        )

    async def _chat_native(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> AsyncIterator[StreamEvent]:
        system, wire = _wire_messages(messages)

        body: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": wire,
            "stream": True,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        # tool_use content blocks accumulate JSON fragments across deltas
        pending: dict[int, dict[str, Any]] = {}

        try:
            async with self._client.stream("POST", "/v1/messages", json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode(errors="replace")[:500]
                    raise ProviderError(
                        f"HTTP {resp.status_code} from Anthropic: {detail}",
                        retryable=resp.status_code >= 500,
                    )
                async for line in resp.aiter_lines():
                    if not line or line.startswith("event:"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    try:
                        chunk = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    etype = chunk.get("type", "")

                    if etype == "message_start":
                        usage = chunk.get("message", {}).get("usage", {})
                        if usage.get("input_tokens"):
                            yield StreamEvent(
                                kind="usage",
                                usage=Usage(
                                    prompt_tokens=usage["input_tokens"],
                                    completion_tokens=0,
                                ),
                            )

                    elif etype == "content_block_start":
                        idx: int = chunk.get("index", 0)
                        block = chunk.get("content_block", {})
                        if block.get("type") == "tool_use":
                            pending[idx] = {
                                "id": block.get("id", f"call_{idx}"),
                                "name": block.get("name", ""),
                                "json_parts": [],
                            }

                    elif etype == "content_block_delta":
                        idx = chunk.get("index", 0)
                        delta = chunk.get("delta", {})
                        dtype = delta.get("type", "")
                        if dtype == "text_delta":
                            yield StreamEvent(kind="text", text=delta.get("text", ""))
                        elif dtype == "input_json_delta" and idx in pending:
                            pending[idx]["json_parts"].append(delta.get("partial_json", ""))

                    elif etype == "content_block_stop":
                        idx = chunk.get("index", 0)
                        if idx in pending:
                            slot = pending.pop(idx)
                            yield StreamEvent(
                                kind="tool_call",
                                tool_call=ToolCall(
                                    id=slot["id"],
                                    name=slot["name"],
                                    arguments=parse_arguments("".join(slot["json_parts"])),
                                ),
                            )

                    elif etype == "message_delta":
                        usage = chunk.get("usage", {})
                        if usage.get("output_tokens"):
                            yield StreamEvent(
                                kind="usage",
                                usage=Usage(
                                    prompt_tokens=0,
                                    completion_tokens=usage["output_tokens"],
                                ),
                            )

        except httpx.HTTPError as e:
            raise ProviderError(
                f"Connection to Anthropic API failed: {e}", retryable=True
            ) from e

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/v1/models")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"Could not list Anthropic models: {e}", retryable=True) from e
        return [m["id"] for m in resp.json().get("data", [])]

    async def close(self) -> None:
        await self._client.aclose()
