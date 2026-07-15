"""Native Ollama provider (POST /api/chat, NDJSON streaming).

Using the native API instead of Ollama's OpenAI shim gives us num_ctx
control, model listing via /api/tags, and keep_alive.
"""

from __future__ import annotations

import json
import uuid
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


def _wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for m in messages:
        entry: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.images and m.role == "user":
            entry["images"] = [img.data for img in m.images]
        if m.tool_calls:
            entry["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}}
                for c in m.tool_calls
            ]
        wire.append(entry)
    return wire


class OllamaProvider(Provider):
    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers=cfg.extra_headers,
            timeout=httpx.Timeout(cfg.timeout_seconds),
        )

    async def _chat_native(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> AsyncIterator[StreamEvent]:
        options: dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "num_ctx": self.cfg.context_length,
            "num_predict": self.cfg.max_tokens,
            **self.cfg.options,
        }
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": _wire_messages(messages),
            "stream": True,
            "options": options,
            "keep_alive": "10m",
        }
        if tools:
            body["tools"] = [t.to_openai() for t in tools]

        try:
            async with self._client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode(errors="replace")[:500]
                    raise ProviderError(
                        f"HTTP {resp.status_code} from Ollama: {detail}",
                        retryable=resp.status_code >= 500,
                    )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        raise ProviderError(f"Ollama: {chunk['error']}")
                    msg = chunk.get("message") or {}
                    if msg.get("content"):
                        yield StreamEvent(kind="text", text=msg["content"])
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        yield StreamEvent(
                            kind="tool_call",
                            tool_call=ToolCall(
                                id=f"call_{uuid.uuid4().hex[:12]}",
                                name=fn.get("name", ""),
                                arguments=parse_arguments(fn.get("arguments")),
                            ),
                        )
                    if chunk.get("done"):
                        yield StreamEvent(
                            kind="usage",
                            usage=Usage(
                                prompt_tokens=chunk.get("prompt_eval_count", 0),
                                completion_tokens=chunk.get("eval_count", 0),
                            ),
                        )
        except httpx.HTTPError as e:
            raise ProviderError(
                f"Connection to Ollama at {self.cfg.base_url} failed: {e}. "
                "Is `ollama serve` running?",
                retryable=True,
            ) from e

    async def pull_model(self, name: str) -> AsyncIterator[str]:
        """Pull a model from the Ollama registry. Yields human-readable status strings."""
        try:
            async with self._client.stream(
                "POST", "/api/pull", json={"model": name, "stream": True}
            ) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode(errors="replace")[:200]
                    raise ProviderError(f"Ollama pull HTTP {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        raise ProviderError(f"Ollama: {chunk['error']}")
                    status = chunk.get("status", "")
                    total = chunk.get("total", 0)
                    if total:
                        completed = chunk.get("completed", 0)
                        pct = round(completed / total * 100)
                        yield f"{status} {pct}%"
                    elif status:
                        yield status
        except httpx.HTTPError as e:
            raise ProviderError(f"Ollama pull failed: {e}", retryable=True) from e

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"Could not list Ollama models: {e}", retryable=True) from e
        return [m["name"] for m in resp.json().get("models", [])]

    async def close(self) -> None:
        await self._client.aclose()
