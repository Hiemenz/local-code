"""Network and clipboard tools: fetch_url, http_request, web_search, clipboard_read/write."""

from __future__ import annotations

import asyncio
import re
import subprocess
from typing import Any
from urllib.parse import unquote

import httpx

from localcode.tools.base import Tool, ToolResult, truncate


def _strip_html(html: str, limit: int) -> str:
    """Strip HTML tags and collapse whitespace to plain text."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return truncate(text.strip(), limit)


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "Fetch a URL and return cleaned text. HTML pages are stripped to readable plain text; "
        "JSON and plain-text responses are returned as-is."
    )
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "timeout": {"type": "number", "description": "Seconds to wait (default 30)"},
        },
        "required": ["url"],
    }

    async def run(self, url: str, timeout: float = 30, **_: Any) -> ToolResult:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                resp = await client.get(url, headers={"User-Agent": "localcode/0.1"})
        except httpx.HTTPError as e:
            return ToolResult.error(str(e))
        ct = resp.headers.get("content-type", "")
        if "html" in ct:
            body = _strip_html(resp.text, self.ctx.max_output_chars())
        else:
            body = truncate(resp.text, self.ctx.max_output_chars())
        return ToolResult(
            ok=True,
            output=f"[{resp.status_code}] {url}\n\n{body}",
            data={"status_code": resp.status_code, "content_type": ct},
        )


class HttpRequestTool(Tool):
    name = "http_request"
    description = (
        "Make an HTTP request (GET/POST/PUT/PATCH/DELETE) and return the status, headers, "
        "and body. Useful for testing local API endpoints during development."
    )
    permission = "execute"
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "default": "GET",
            },
            "url": {"type": "string"},
            "headers": {
                "type": "object",
                "description": "HTTP headers as string key-value pairs",
            },
            "body": {"type": "string", "description": "Request body (JSON or raw text)"},
            "timeout": {"type": "number", "description": "Seconds to wait (default 30)"},
        },
        "required": ["url"],
    }

    async def run(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: str | None = None,
        timeout: float = 30,
        **_: Any,
    ) -> ToolResult:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                resp = await client.request(
                    method.upper(),
                    url,
                    headers=headers or {},
                    content=body.encode() if body else None,
                )
        except httpx.HTTPError as e:
            return ToolResult.error(str(e))
        header_block = "\n".join(f"{k}: {v}" for k, v in resp.headers.items())
        output = (
            f"HTTP {resp.status_code} {resp.reason_phrase}\n"
            f"{header_block}\n\n"
            f"{truncate(resp.text, self.ctx.max_output_chars())}"
        )
        return ToolResult(
            ok=200 <= resp.status_code < 300,
            output=output,
            data={"status_code": resp.status_code},
        )

    def preview(self, args: dict[str, Any]) -> str:
        return f"{args.get('method', 'GET')} {args.get('url', '')}"


def _try_clipboard_read() -> str:
    for cmd in (
        ["xclip", "-o", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--output"],
        ["wl-paste"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=5)
            if r.returncode == 0:
                return r.stdout.decode(errors="replace")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise OSError("No clipboard tool found (install xclip, xsel, or wl-clipboard)")


def _try_clipboard_write(text: str) -> None:
    for cmd in (
        ["xclip", "-i", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["wl-copy"],
    ):
        try:
            r = subprocess.run(cmd, input=text.encode(), capture_output=True, timeout=5)
            if r.returncode == 0:
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise OSError("No clipboard tool found (install xclip, xsel, or wl-clipboard)")


class ClipboardReadTool(Tool):
    name = "clipboard_read"
    description = (
        "Read the current contents of the system clipboard. "
        "Linux: requires xclip, xsel, or wl-paste (Wayland)."
    )
    permission = "read"
    parameters = {"type": "object", "properties": {}}

    async def run(self, **_: Any) -> ToolResult:
        try:
            text = await asyncio.to_thread(_try_clipboard_read)
        except OSError as e:
            return ToolResult.error(str(e))
        return ToolResult(ok=True, output=truncate(text, self.ctx.max_output_chars()))


class ClipboardWriteTool(Tool):
    name = "clipboard_write"
    description = (
        "Write text to the system clipboard. "
        "Linux: requires xclip, xsel, or wl-copy (Wayland)."
    )
    permission = "write"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, text: str, **_: Any) -> ToolResult:
        try:
            await asyncio.to_thread(_try_clipboard_write, text)
        except OSError as e:
            return ToolResult.error(str(e))
        return ToolResult(ok=True, output=f"Wrote {len(text)} chars to clipboard.")


def _parse_ddg_results(html: str, limit: int) -> list[dict[str, str]]:
    """Extract result title/URL/snippet from DuckDuckGo HTML."""
    results: list[dict[str, str]] = []
    # Each organic result sits inside a div whose class starts with "result "
    blocks = re.split(r'<div[^>]+class="result(?:\s[^"]*)?web-result[^"]*"', html)
    for block in blocks[1:]:
        title_m = re.search(r'class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL)
        if not title_m:
            continue
        title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()

        # DuckDuckGo wraps the real URL in a redirect; extract from uddg param
        uddg_m = re.search(r'uddg=([^&"]+)', block)
        if uddg_m:
            url = unquote(uddg_m.group(1))
        else:
            href_m = re.search(r'class="result__a"[^>]*href="([^"]*)"', block)
            url = href_m.group(1) if href_m else ""

        snippet_m = re.search(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>', block, re.DOTALL
        )
        snippet = re.sub(r"<[^>]+>", "", snippet_m.group(1)).strip() if snippet_m else ""

        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web via DuckDuckGo and return titles, URLs, and snippets. "
        "No API key required. Use for current events, documentation lookups, or "
        "any question that benefits from live web results."
    )
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1–10, default 5)",
            },
        },
        "required": ["query"],
    }

    async def run(self, query: str, max_results: int = 5, **_: Any) -> ToolResult:
        max_results = max(1, min(max_results, 10))
        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=30.0
            ) as client:
                resp = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
                            "Gecko/20100101 Firefox/120.0"
                        ),
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "text/html",
                    },
                )
        except httpx.HTTPError as e:
            return ToolResult.error(f"Search request failed: {e}")

        results = _parse_ddg_results(resp.text, max_results)
        if not results:
            return ToolResult(
                ok=True,
                output=f"No results found for: {query}",
                data={"query": query, "results": []},
            )

        lines: list[str] = [f"Web search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['url']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet']}")
            lines.append("")

        return ToolResult(
            ok=True,
            output=truncate("\n".join(lines).strip(), self.ctx.max_output_chars()),
            data={"query": query, "results": results},
        )


TOOLS = [FetchUrlTool, HttpRequestTool, WebSearchTool, ClipboardReadTool, ClipboardWriteTool]
