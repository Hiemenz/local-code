import json

import httpx
import pytest

from localcode.config.schema import ProviderConfig
from localcode.errors import ProviderError
from localcode.providers.anthropic import AnthropicProvider, _wire_messages
from localcode.providers.base import ImageContent, Message, ToolCall, ToolSpec, create_provider
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


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

def anthropic_sse(*chunks: dict) -> bytes:
    """Build an Anthropic-style SSE body from a sequence of event dicts."""
    lines: list[str] = []
    for chunk in chunks:
        lines.append(f"event: {chunk.get('type', 'data')}")
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    return "\n".join(lines).encode()


def anthropic_provider(handler) -> AnthropicProvider:
    cfg = ProviderConfig(
        type="anthropic", base_url="http://test", model="claude-opus-4-8", api_key="sk-test"
    )
    provider = AnthropicProvider(cfg)
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )
    return provider


async def test_anthropic_streams_text_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        body = json.loads(request.content)
        assert body["model"] == "claude-opus-4-8"
        assert body["stream"] is True
        return httpx.Response(200, content=anthropic_sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 12, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ))

    events = await drain(anthropic_provider(handler), MSGS)
    text = "".join(e.text for e in events if e.kind == "text")
    assert text == "Hello"
    usage_events = [e.usage for e in events if e.kind == "usage" and e.usage is not None]
    assert any(u.prompt_tokens == 12 for u in usage_events)
    assert any(u.completion_tokens == 5 for u in usage_events)


async def test_anthropic_streams_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"][0]["input_schema"] == SPEC.parameters
        return httpx.Response(200, content=anthropic_sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 8, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "toolu_01", "name": "read_file"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": '{"path"'}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": ': "a.py"}'}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 3}},
        ))

    events = await drain(anthropic_provider(handler), MSGS, [SPEC])
    calls = [e.tool_call for e in events if e.kind == "tool_call"]
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}
    assert calls[0].id == "toolu_01"


async def test_anthropic_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(529, content=b'{"type":"error","error":{"message":"overloaded"}}')

    with pytest.raises(ProviderError) as info:
        await drain(anthropic_provider(handler), MSGS)
    assert info.value.retryable
    assert "529" in str(info.value)


async def test_anthropic_list_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={
            "data": [{"id": "claude-opus-4-8"}, {"id": "claude-sonnet-5"}],
            "has_more": False,
        })

    models = await anthropic_provider(handler).list_models()
    assert models == ["claude-opus-4-8", "claude-sonnet-5"]


def test_anthropic_added_to_factory() -> None:
    cfg = ProviderConfig(type="anthropic", api_key="sk-test")
    assert isinstance(create_provider(cfg), AnthropicProvider)


# -- _wire_messages unit tests ------------------------------------------------

def test_wire_extracts_system_prompt() -> None:
    msgs = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="hi"),
    ]
    system, wire = _wire_messages(msgs)
    assert system == "You are helpful."
    assert len(wire) == 1
    assert wire[0]["role"] == "user"


def test_wire_batches_consecutive_tool_results() -> None:
    msgs = [
        Message(role="user", content="do both"),
        Message(role="assistant", content="", tool_calls=[
            ToolCall(id="c1", name="read_file", arguments={"path": "a.py"}),
            ToolCall(id="c2", name="read_file", arguments={"path": "b.py"}),
        ]),
        Message(role="tool", content="contents a", tool_call_id="c1", name="read_file"),
        Message(role="tool", content="contents b", tool_call_id="c2", name="read_file"),
    ]
    _, wire = _wire_messages(msgs)
    # user, assistant, user(batched tool results)
    assert len(wire) == 3
    last = wire[-1]
    assert last["role"] == "user"
    assert len(last["content"]) == 2
    assert last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "c1"
    assert last["content"][1]["tool_use_id"] == "c2"


def test_wire_assistant_tool_calls_become_tool_use_blocks() -> None:
    msgs = [
        Message(role="assistant", content="I'll read it.", tool_calls=[
            ToolCall(id="c1", name="read_file", arguments={"path": "a.py"}),
        ]),
    ]
    _, wire = _wire_messages(msgs)
    assert wire[0]["role"] == "assistant"
    content = wire[0]["content"]
    assert content[0] == {"type": "text", "text": "I'll read it."}
    assert content[1]["type"] == "tool_use"
    assert content[1]["id"] == "c1"
    assert content[1]["input"] == {"path": "a.py"}


def test_wire_image_in_user_message() -> None:
    msgs = [
        Message(role="user", content="what's this?",
                images=[ImageContent(data="abc123", mime_type="image/png")]),
    ]
    _, wire = _wire_messages(msgs)
    content = wire[0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["data"] == "abc123"
    assert content[1]["type"] == "text"


# ---------------------------------------------------------------------------
# Ollama pull_model
# ---------------------------------------------------------------------------

async def test_ollama_pull_model_yields_status() -> None:
    lines = [
        {"status": "pulling manifest"},
        {"status": "downloading", "completed": 500, "total": 1000},
        {"status": "downloading", "completed": 1000, "total": 1000},
        {"status": "success"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/pull"
        body = json.loads(request.content)
        assert body["model"] == "llama3.2"
        content = "".join(json.dumps(line) + "\n" for line in lines)
        return httpx.Response(200, content=content.encode())

    statuses = [s async for s in ollama_provider(handler).pull_model("llama3.2")]
    assert statuses[0] == "pulling manifest"
    assert "50%" in statuses[1]
    assert "100%" in statuses[2]
    assert statuses[-1] == "success"


async def test_ollama_pull_model_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"error": "model not found"}).encode())

    with pytest.raises(ProviderError, match="model not found"):
        async for _ in ollama_provider(handler).pull_model("nope"):
            pass
