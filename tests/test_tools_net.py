"""Tests for net tools: FetchUrlTool, HttpRequestTool, ClipboardRead/Write."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from localcode.tools.base import ToolContext
from localcode.tools.net import (
    ClipboardReadTool,
    ClipboardWriteTool,
    FetchUrlTool,
    HttpRequestTool,
    _strip_html,
)

# -- _strip_html helper -------------------------------------------------------

def test_strip_html_removes_tags() -> None:
    html = "<html><body><p>Hello <b>world</b></p></body></html>"
    result = _strip_html(html, 10_000)
    assert "Hello" in result
    assert "world" in result
    assert "<" not in result


def test_strip_html_removes_scripts_and_styles() -> None:
    html = "<script>alert(1)</script><style>.a{}</style><p>Clean</p>"
    result = _strip_html(html, 10_000)
    assert "alert" not in result
    assert ".a{}" not in result
    assert "Clean" in result


def test_strip_html_collapses_whitespace() -> None:
    html = "<p>Hello</p>   <p>World</p>"
    result = _strip_html(html, 10_000)
    assert "\n\n\n" not in result


# -- FetchUrlTool ---------------------------------------------------------------

def _mock_html_response(status: int = 200, html: str = "<p>Hello</p>") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.reason_phrase = "OK"
    resp.headers = {"content-type": "text/html; charset=utf-8"}
    resp.text = html
    return resp


def _mock_json_response(body: str = '{"ok": true}') -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.reason_phrase = "OK"
    resp.headers = {"content-type": "application/json"}
    resp.text = body
    return resp


async def test_fetch_url_html_strips_tags(tool_ctx: ToolContext) -> None:
    resp = _mock_html_response(html="<html><body><p>Hello world</p></body></html>")
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_cls.return_value)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value.get = AsyncMock(return_value=resp)
        result = await FetchUrlTool(tool_ctx).run(url="http://example.com")
    assert result.ok
    assert "Hello world" in result.output
    assert "<" not in result.output


async def test_fetch_url_json_not_stripped(tool_ctx: ToolContext) -> None:
    resp = _mock_json_response('{"key": "value"}')
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_cls.return_value)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value.get = AsyncMock(return_value=resp)
        result = await FetchUrlTool(tool_ctx).run(url="http://api.example.com/data")
    assert result.ok
    assert '"key"' in result.output


async def test_fetch_url_http_error(tool_ctx: ToolContext) -> None:
    import httpx

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_cls.return_value)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        result = await FetchUrlTool(tool_ctx).run(url="http://localhost:9999")
    assert not result.ok
    assert "Connection refused" in result.output


# -- HttpRequestTool -----------------------------------------------------------

async def test_http_request_get(tool_ctx: ToolContext) -> None:
    resp = MagicMock()
    resp.status_code = 200
    resp.reason_phrase = "OK"
    resp.headers = {"content-type": "text/plain"}
    resp.text = "pong"
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_cls.return_value)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value.request = AsyncMock(return_value=resp)
        result = await HttpRequestTool(tool_ctx).run(url="http://localhost/ping")
    assert result.ok
    assert "200" in result.output
    assert "pong" in result.output


async def test_http_request_non_2xx_is_not_ok(tool_ctx: ToolContext) -> None:
    resp = MagicMock()
    resp.status_code = 404
    resp.reason_phrase = "Not Found"
    resp.headers = {}
    resp.text = "not found"
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_cls.return_value)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value.request = AsyncMock(return_value=resp)
        result = await HttpRequestTool(tool_ctx).run(url="http://localhost/nope", method="GET")
    assert not result.ok
    assert "404" in result.output


async def test_http_request_preview(tool_ctx: ToolContext) -> None:
    tool = HttpRequestTool(tool_ctx)
    assert tool.preview({"method": "POST", "url": "http://a.com"}) == "POST http://a.com"


# -- ClipboardReadTool ----------------------------------------------------------

async def test_clipboard_read_xclip(tool_ctx: ToolContext) -> None:
    mock_result = MagicMock(returncode=0, stdout=b"hello from clipboard")
    with patch("subprocess.run", return_value=mock_result):
        result = await ClipboardReadTool(tool_ctx).run()
    assert result.ok
    assert "hello from clipboard" in result.output


async def test_clipboard_read_no_tool(tool_ctx: ToolContext) -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = await ClipboardReadTool(tool_ctx).run()
    assert not result.ok
    assert "No clipboard tool" in result.output


# -- ClipboardWriteTool ---------------------------------------------------------

async def test_clipboard_write_xclip(tool_ctx: ToolContext) -> None:
    mock_result = MagicMock(returncode=0)
    with patch("subprocess.run", return_value=mock_result):
        result = await ClipboardWriteTool(tool_ctx).run(text="hello")
    assert result.ok
    assert "5 chars" in result.output


async def test_clipboard_write_no_tool(tool_ctx: ToolContext) -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = await ClipboardWriteTool(tool_ctx).run(text="hi")
    assert not result.ok
    assert "No clipboard tool" in result.output
