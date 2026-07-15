"""Targeted tests for the remaining under-covered modules:
repo tools with a live index, plugin module-path loading, build/node tools."""

from __future__ import annotations

import shutil
from typing import Any

import pytest

from localcode.config.schema import IndexConfig
from localcode.index.repomap import RepoIndex
from localcode.tools import shell
from localcode.tools.base import ToolContext, ToolResult, build_default_registry
from localcode.tools.repo import FindRelevantFilesTool, RepoMapTool
from localcode.tools.shell import RunBuildTool, RunNodeTool


@pytest.fixture
def indexed_ctx(tool_ctx: ToolContext) -> ToolContext:
    index = RepoIndex(tool_ctx.root, IndexConfig(), None)
    index.scan()
    tool_ctx.index = index
    return tool_ctx


# -- repo tools -----------------------------------------------------------------

async def test_repo_map_tool(indexed_ctx: ToolContext) -> None:
    result = await RepoMapTool(indexed_ctx).run()
    assert result.ok
    assert "src/util.py" in result.output
    assert "python" in result.output


async def test_repo_map_tool_without_index(tool_ctx: ToolContext) -> None:
    result = await RepoMapTool(tool_ctx).run()
    assert not result.ok


async def test_find_relevant_files_tool(indexed_ctx: ToolContext) -> None:
    result = await FindRelevantFilesTool(indexed_ctx).run(query="helper util")
    assert result.ok
    assert "src/util.py" in result.output
    assert result.data is not None and "src/util.py" in result.data["files"]


async def test_find_relevant_files_no_match(indexed_ctx: ToolContext) -> None:
    result = await FindRelevantFilesTool(indexed_ctx).run(query="zzznotfound")
    assert result.ok
    assert "No relevant files" in result.output


async def test_find_relevant_files_without_index(tool_ctx: ToolContext) -> None:
    result = await FindRelevantFilesTool(tool_ctx).run(query="x")
    assert not result.ok


# -- plugin module-path loading ----------------------------------------------------

def test_config_module_plugin(tool_ctx: ToolContext, tmp_path_factory: pytest.TempPathFactory,
                              monkeypatch: pytest.MonkeyPatch) -> None:
    plugin_dir = tmp_path_factory.mktemp("pluginmod")
    (plugin_dir / "my_plugin_mod.py").write_text(
        "from localcode.tools.base import Tool, ToolResult\n"
        "class PingTool(Tool):\n"
        "    name = 'ping'\n"
        "    description = 'Ping.'\n"
        "    permission = 'read'\n"
        "    parameters = {'type': 'object', 'properties': {}}\n"
        "    async def run(self, **kwargs):\n"
        "        return ToolResult(ok=True, output='pong')\n"
        "def register(registry, ctx):\n"
        "    registry.register(PingTool(ctx))\n"
    )
    monkeypatch.syspath_prepend(str(plugin_dir))
    tool_ctx.config.plugins = ["my_plugin_mod"]
    registry = build_default_registry(tool_ctx)
    assert registry.get("ping") is not None


def test_missing_module_plugin_is_skipped(tool_ctx: ToolContext) -> None:
    tool_ctx.config.plugins = ["does.not.exist"]
    registry = build_default_registry(tool_ctx)  # must not raise
    assert registry.get("read_file") is not None


def test_module_plugin_without_register(tool_ctx: ToolContext,
                                        tmp_path_factory: pytest.TempPathFactory,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    plugin_dir = tmp_path_factory.mktemp("pluginmod2")
    (plugin_dir / "no_register_mod.py").write_text("x = 1\n")
    monkeypatch.syspath_prepend(str(plugin_dir))
    tool_ctx.config.plugins = ["no_register_mod"]
    registry = build_default_registry(tool_ctx)  # warns, does not raise
    assert registry.get("read_file") is not None


# -- build / node tools ----------------------------------------------------------------

async def test_run_build_no_build_system(tool_ctx: ToolContext) -> None:
    result = await RunBuildTool(tool_ctx).run()
    assert not result.ok
    assert "No build system" in result.output


async def test_run_build_uses_detected_command(tool_ctx: ToolContext,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    (tool_ctx.root / "Makefile").write_text("all:\n\t@echo built\n")
    calls: list[str] = []

    async def fake_run(command: str, cwd: str, timeout: float, limit: int) -> ToolResult:
        calls.append(command)
        return ToolResult(ok=True, output="built")

    monkeypatch.setattr(shell, "run_subprocess", fake_run)
    result = await RunBuildTool(tool_ctx).run(args="-j2")
    assert result.ok
    assert calls == ["make -j2"]
    assert result.output.startswith("$ make -j2")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
async def test_run_node(tool_ctx: ToolContext) -> None:
    result = await RunNodeTool(tool_ctx).run(code="console.log(1 + 1)")
    assert result.ok and "2" in result.output


async def test_run_node_missing_binary(tool_ctx: ToolContext,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(command: str, cwd: str, timeout: float, limit: int) -> ToolResult:
        return ToolResult(ok=False, output="[exit code 127]\nnode: command not found")

    monkeypatch.setattr(shell, "run_subprocess", fake_run)
    result = await RunNodeTool(tool_ctx).run(code="1")
    assert not result.ok


# -- tool crash isolation (agent-side safety net) ---------------------------------------

async def test_tool_crash_becomes_error_result(tool_ctx, config, fake_provider_factory) -> None:
    from localcode.agent.loop import Agent
    from localcode.providers.base import ToolCall
    from localcode.tools.base import Tool

    class CrashTool(Tool):
        name = "crash"
        description = "Crashes."
        permission = "read"
        parameters = {"type": "object", "properties": {}}

        async def run(self, **kwargs: Any) -> ToolResult:
            raise RuntimeError("kaboom")

    config.agent.auto_test = False
    registry = build_default_registry(tool_ctx)
    registry.register(CrashTool(tool_ctx))
    provider = fake_provider_factory(
        [("", [ToolCall(id="1", name="crash", arguments={})]), ("recovered", [])]
    )
    agent = Agent(provider=provider, registry=registry, tool_ctx=tool_ctx, config=config)
    agent.set_system_prompt("t")
    events = [e async for e in agent.run_turn("go")]
    ends = [e for e in events if e.kind == "tool_end"]
    assert not ends[0].result.ok
    assert "kaboom" in ends[0].result.output
    assert agent.messages[-1].content == "recovered"  # loop survived


async def test_invalid_tool_arguments_reported(tool_ctx, config, fake_provider_factory) -> None:
    from localcode.agent.loop import Agent
    from localcode.providers.base import ToolCall

    config.agent.auto_test = False
    registry = build_default_registry(tool_ctx)
    provider = fake_provider_factory(
        [("", [ToolCall(id="1", name="read_file", arguments={"bogus_arg": 1})]), ("ok", [])]
    )
    agent = Agent(provider=provider, registry=registry, tool_ctx=tool_ctx, config=config)
    agent.set_system_prompt("t")
    events = [e async for e in agent.run_turn("go")]
    ends = [e for e in events if e.kind == "tool_end"]
    assert not ends[0].result.ok
    assert "Invalid arguments" in ends[0].result.output
