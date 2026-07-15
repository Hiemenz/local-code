"""Tests for McpStdioClient and load_mcp_tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from localcode.mcp.client import McpStdioClient, McpTool, load_mcp_tools
from localcode.tools.base import ToolContext, ToolRegistry

# A minimal MCP server script written inline as a string.
_MCP_SERVER = """
import sys, json, uuid

def reply(rid, result):
    print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    rid = msg["id"]
    method = msg["method"]
    if method == "initialize":
        reply(rid, {"protocolVersion": "2024-11-05", "capabilities": {}})
    elif method == "tools/list":
        reply(rid, {"tools": [
            {"name": "echo", "description": "Echo the input", "inputSchema": {
                "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]
            }}
        ]})
    elif method == "tools/call":
        args = msg["params"].get("arguments", {})
        reply(rid, {"content": [{"type": "text", "text": args.get("text", "")}]})
"""


async def test_mcp_list_and_call_tools() -> None:
    client = McpStdioClient(sys.executable, ["-c", _MCP_SERVER])
    try:
        await client.start()
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"
        output = await client.call_tool("echo", {"text": "hello MCP"})
        assert output == "hello MCP"
    finally:
        await client.close()


async def test_load_mcp_tools_from_config(tool_ctx: ToolContext, tmp_path: Path) -> None:
    mcp_json = tool_ctx.root / ".mcp.json"
    mcp_json.write_text(
        json.dumps({
            "mcpServers": {
                "test": {
                    "command": sys.executable,
                    "args": ["-c", _MCP_SERVER],
                }
            }
        }),
        encoding="utf-8",
    )
    registry = ToolRegistry()
    await load_mcp_tools(registry, tool_ctx, tool_ctx.root)
    names = registry.names()
    assert "mcp_test_echo" in names


async def test_load_mcp_tools_no_config(tool_ctx: ToolContext) -> None:
    """load_mcp_tools should silently do nothing when .mcp.json is absent."""
    registry = ToolRegistry()
    await load_mcp_tools(registry, tool_ctx, tool_ctx.root)
    assert registry.names() == []


async def test_load_mcp_tools_bad_json(tool_ctx: ToolContext) -> None:
    (tool_ctx.root / ".mcp.json").write_text("not json", encoding="utf-8")
    registry = ToolRegistry()
    await load_mcp_tools(registry, tool_ctx, tool_ctx.root)  # must not raise
    assert registry.names() == []


async def test_mcp_tool_run(tool_ctx: ToolContext) -> None:
    client = McpStdioClient(sys.executable, ["-c", _MCP_SERVER])
    await client.start()
    try:
        tool_def = {
            "name": "echo",
            "description": "Echo it",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
        }
        tool = McpTool(tool_ctx, client, "myserver", tool_def)
        assert tool.name == "mcp_myserver_echo"
        result = await tool.run(text="hi there")
        assert result.ok
        assert "hi there" in result.output
    finally:
        await client.close()
