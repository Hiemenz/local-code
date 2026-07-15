import json

import httpx
import pytest

from localcode.config.schema import ProviderConfig
from localcode.errors import ProviderError
from localcode.providers.base import Message, ToolSpec, create_provider
from localcode.providers.ollama import OllamaProvider
from localcode.providers.openai_compat import OpenAICompatProvider

SPEC = ToolSpec(name="read_file", description="d", parameters={"type": "object"})
MSGS = [Message(role="user", content="hi")]


def sse(chunks: list[dict]) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


def openai_provider(handler) -> OpenAICompatProvider:
    cfg = ProviderConfig(type="openai", base_url="http://test", model="m")
    provider = OpenAICompatProvider(cfg)
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test/v1"
    )
    return provider


def ollama_provider(handler) -> OllamaProvider:
    cfg = ProviderConfig(type="ollama", base_url="http://test", model="m")
    provider = OllamaProvider(cfg)
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )
    return provider


async def drain(provider, messages, tools=None):
    return [event async for event in provider.chat(messages, tools)]


async def test_openai_streams_text_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse([
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        ]))

    events = await drain(openai_provider(handler), MSGS)
    text = "".join(e.text for e in events if e.kind == "text")
    assert text == "Hello"
    usage = [e for e in events if e.kind == "usage"][0].usage
    assert usage.prompt_tokens == 3


async def test_openai_assembles_tool_call_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "read_", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "file", "arguments": '{"path"'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ': "a.py"}'}}]}}]},
        ]))

    events = await drain(openai_provider(handler), MSGS, [SPEC])
    calls = [e.tool_call for e in events if e.kind == "tool_call"]
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}


async def test_openai_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    with pytest.raises(ProviderError) as info:
        await drain(openai_provider(handler), MSGS)
    assert info.value.retryable


async def test_auto_mode_falls_back_to_json_tools() -> None:
    """A server that 400s on native tools should trigger the <tool_call> fallback."""
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_bodies.append(body)
        if "tools" in body:
            return httpx.Response(400, content=b"tools not supported")
        return httpx.Response(200, content=sse([
            {"choices": [{"delta": {"content":
                'Sure. <tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'}}]},
        ]))

    provider = openai_provider(handler)
    events = await drain(provider, MSGS, [SPEC])
    calls = [e.tool_call for e in events if e.kind == "tool_call"]
    assert len(calls) == 1 and calls[0].arguments == {"path": "a.py"}
    text = "".join(e.text for e in events if e.kind == "text")
    assert "<tool_call>" not in text
    # second request used prompt-injected tools, not the tools param
    assert "tools" not in seen_bodies[-1]
    assert "read_file" in seen_bodies[-1]["messages"][0]["content"]
    # provider remembers and skips native next time
    assert provider._native_tools_broken


async def test_json_mode_skips_native_tools() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tools" not in body
        return httpx.Response(200, content=sse([
            {"choices": [{"delta": {"content": "plain answer"}}]},
        ]))

    cfg = ProviderConfig(type="openai", base_url="http://test", model="m", tool_mode="json")
    provider = OpenAICompatProvider(cfg)
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test/v1"
    )
    events = await drain(provider, MSGS, [SPEC])
    assert "".join(e.text for e in events if e.kind == "text") == "plain answer"


async def test_ollama_ndjson_stream() -> None:
    lines = [
        {"message": {"content": "Hi "}},
        {"message": {"content": "there"}},
        {"message": {"tool_calls": [
            {"function": {"name": "grep", "arguments": {"pattern": "x"}}}]}},
        {"done": True, "prompt_eval_count": 7, "eval_count": 4},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["options"]["num_ctx"] == 32768
        content = "".join(json.dumps(line) + "\n" for line in lines)
        return httpx.Response(200, content=content.encode())

    events = await drain(ollama_provider(handler), MSGS, [SPEC])
    assert "".join(e.text for e in events if e.kind == "text") == "Hi there"
    calls = [e.tool_call for e in events if e.kind == "tool_call"]
    assert calls[0].name == "grep" and calls[0].arguments == {"pattern": "x"}
    usage = [e for e in events if e.kind == "usage"][0].usage
    assert usage.prompt_tokens == 7 and usage.completion_tokens == 4


async def test_ollama_error_line_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"error": "model not found"}).encode())

    with pytest.raises(ProviderError) as info:
        await drain(ollama_provider(handler), MSGS)
    assert "model not found" in str(info.value)


async def test_ollama_list_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5-coder:14b"}]})

    models = await ollama_provider(handler).list_models()
    assert models == ["qwen2.5-coder:14b"]


def test_create_provider_factory() -> None:
    assert isinstance(create_provider(ProviderConfig(type="ollama")), OllamaProvider)
    assert isinstance(create_provider(ProviderConfig(type="openai")), OpenAICompatProvider)


async def test_complete_non_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse([
            {"choices": [{"delta": {"content": "summary text"}}]},
        ]))

    result = await openai_provider(handler).complete(MSGS)
    assert result == "summary text"
