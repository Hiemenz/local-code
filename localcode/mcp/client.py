"""MCP stdio client: JSON-RPC 2.0 communication with MCP servers over subprocess stdio.

Reads .mcp.json from the project root (same format as Claude Code):
  {
    "mcpServers": {
      "my-server": {
        "command": "python3",
        "args": ["/path/to/server.py"],
        "env": {"API_KEY": "..."}
      }
    }
  }
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from localcode.logging import get_logger
from localcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult

log = get_logger("mcp")


class McpStdioClient:
    """Async JSON-RPC 2.0 client speaking to an MCP server over its stdin/stdout."""

    def __init__(
        self, command: str, args: list[str], env: dict[str, str] | None = None
    ) -> None:
        self.command = command
        self.args = args
        self.env = env
        self._proc: asyncio.subprocess.Process | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        merged: dict[str, str] = dict(os.environ)
        if self.env:
            merged.update(self.env)
        self._proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=merged,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "localcode", "version": "0.1"},
            },
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._request("tools/list", {})
        raw = result.get("tools", [])
        return raw if isinstance(raw, list) else []  # type: ignore[no-any-return]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(parts)

    async def close(self) -> None:
        self._closed = True
        if self._proc and self._proc.stdin:
            with contextlib.suppress(Exception):
                self._proc.stdin.close()
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc:
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                self._proc.kill()

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Client not started")
        rid = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[rid] = future
        msg = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        self._proc.stdin.write((msg + "\n").encode())
        await self._proc.stdin.drain()
        return await asyncio.wait_for(future, timeout=30)

    async def _read_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        while not self._closed:
            try:
                line = await self._proc.stdout.readline()
            except Exception:  # noqa: BLE001
                break
            if not line:
                break
            try:
                msg = json.loads(line)
                rid = str(msg.get("id", ""))
                if rid in self._pending:
                    fut = self._pending.pop(rid)
                    if not fut.done():
                        if "error" in msg:
                            fut.set_exception(RuntimeError(str(msg["error"])))
                        else:
                            fut.set_result(msg.get("result", {}))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass


class McpTool(Tool):
    """A local Tool that delegates execution to a remote MCP server tool."""

    def __init__(
        self,
        ctx: ToolContext,
        client: McpStdioClient,
        server_name: str,
        tool_def: dict[str, Any],
    ) -> None:
        super().__init__(ctx)
        raw_name = str(tool_def.get("name", "unknown"))
        self.name = f"mcp_{server_name}_{raw_name}"
        self.description = str(
            tool_def.get("description", f"MCP tool '{raw_name}' from server '{server_name}'")
        )
        self.parameters: dict[str, Any] = tool_def.get(
            "inputSchema", {"type": "object", "properties": {}}
        )
        self.permission = "execute"
        self._client = client
        self._tool_name = raw_name

    async def run(self, **kwargs: Any) -> ToolResult:
        try:
            output = await self._client.call_tool(self._tool_name, kwargs)
            return ToolResult(ok=True, output=output or "(no output)")
        except Exception as e:  # noqa: BLE001
            return ToolResult.error(str(e))

    def preview(self, args: dict[str, Any]) -> str:
        return f"mcp:{self._tool_name}({', '.join(f'{k}=...' for k in args)})"


async def load_mcp_tools(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    """Read .mcp.json from project_root and register each server's tools."""
    mcp_config_path = project_root / ".mcp.json"
    if not mcp_config_path.exists():
        return
    try:
        config = json.loads(mcp_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"Could not read .mcp.json: {e}")
        return
    for name, server_cfg in config.get("mcpServers", {}).items():
        command = server_cfg.get("command", "")
        args: list[str] = server_cfg.get("args", [])
        env: dict[str, str] | None = server_cfg.get("env")
        if not command:
            log.warning(f"MCP server '{name}' has no command, skipping")
            continue
        client = McpStdioClient(command, args, env)
        try:
            await asyncio.wait_for(client.start(), timeout=10)
            tools = await asyncio.wait_for(client.list_tools(), timeout=10)
            for tool_def in tools:
                registry.register(McpTool(ctx, client, name, tool_def))
            log.info(f"MCP server '{name}': registered {len(tools)} tool(s)")
        except Exception as e:  # noqa: BLE001
            log.warning(f"MCP server '{name}' failed to start: {e}")
            with contextlib.suppress(Exception):
                await client.close()
