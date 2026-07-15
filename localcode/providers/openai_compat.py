"""OpenAI-compatible chat completions provider.

Works with LM Studio, vLLM, llama.cpp server (llama-server), text-generation-
webui, and any other server exposing POST /v1/chat/completions with SSE.
"""

from __future__ import annotations

import json
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
        if m.images and m.role == "user":
            content: str | list[dict[str, Any]] = [
                {"type": "text", "text": m.content},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img.mime_type};base64,{img.data}"},
                    }
                    for img in m.images
                ],
            ]
        else:
            content = m.content
        entry: dict[str, Any] = {"role": m.role, "content": content}
        if m.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in m.tool_calls
            ]
        if m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        if m.name and m.role == "tool":
            entry["name"] = m.name
        wire.append(entry)
    return wire


class OpenAICompatProvider(Provider):
    def __init__(self, cfg: ProviderConfig) -> None:
        super().__init__(cfg)
        base = cfg.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        headers = {"Content-Type": "application/json", **cfg.extra_headers}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self._client = httpx.AsyncClient(
            base_url=base, headers=headers, timeout=httpx.Timeout(cfg.timeout_seconds)
        )

    async def _chat_native(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> AsyncIterator[StreamEvent]:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": _wire_messages(messages),
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.cfg.options,
        }
        if tools:
            body["tools"] = [t.to_openai() for t in tools]

        # id/name/argument-fragments accumulated per tool_call index
        pending: dict[int, dict[str, Any]] = {}
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode(errors="replace")[:500]
                    raise ProviderError(
                        f"HTTP {resp.status_code} from {self.cfg.base_url}: {detail}",
                        retryable=resp.status_code >= 500,
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage")
                    if usage:
                        yield StreamEvent(
                            kind="usage",
                            usage=Usage(
                                prompt_tokens=usage.get("prompt_tokens", 0),
                                completion_tokens=usage.get("completion_tokens", 0),
                            ),
                        )
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield StreamEvent(kind="text", text=content)
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
        except httpx.HTTPError as e:
            raise ProviderError(f"Connection to {self.cfg.base_url} failed: {e}", retryable=True) from e

        for i, slot in sorted(pending.items()):
            yield StreamEvent(
                kind="tool_call",
                tool_call=ToolCall(
                    id=slot["id"] or f"call_{i}",
                    name=slot["name"],
                    arguments=parse_arguments(slot["args"]),
                ),
            )

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/models")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"Could not list models: {e}", retryable=True) from e
        return [m["id"] for m in resp.json().get("data", [])]

    async def close(self) -> None:
        await self._client.aclose()
