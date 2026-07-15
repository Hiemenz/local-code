from pathlib import Path

import pytest

from localcode.tools.base import ToolContext, build_default_registry


def test_default_registry_has_all_builtins(tool_ctx: ToolContext) -> None:
    registry = build_default_registry(tool_ctx)
    names = set(registry.names())
    expected = {
        "read_file", "write_file", "edit_file", "apply_patch", "list_dir", "glob",
        "grep", "find_files", "bash", "run_python", "run_node", "run_tests", "run_build",
        "git_status", "git_diff", "git_commit", "git_branch", "git_log",
        "rename_symbol", "repo_map", "find_relevant_files",
    }
    assert expected <= names


def test_all_tools_have_json_schemas(tool_ctx: ToolContext) -> None:
    registry = build_default_registry(tool_ctx)
    for spec in registry.specs():
        assert spec.parameters.get("type") == "object", spec.name
        assert spec.description


def test_file_plugin_loading(tool_ctx: ToolContext, tmp_path: Path,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    plugins_dir = tmp_path / "cfg" / "localcode" / "plugins"
    plugins_dir.mkdir(parents=True)
    (plugins_dir / "hello.py").write_text(
        '''
from localcode.tools.base import Tool, ToolResult

class HelloTool(Tool):
    name = "hello"
    description = "Say hello"
    permission = "read"
    parameters = {"type": "object", "properties": {}}

    async def run(self, **kwargs):
        return ToolResult(ok=True, output="hello from plugin")

def register(registry, ctx):
    registry.register(HelloTool(ctx))
'''
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    registry = build_default_registry(tool_ctx)
    assert registry.get("hello") is not None


def test_broken_plugin_does_not_crash(tool_ctx: ToolContext, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    plugins_dir = tmp_path / "cfg" / "localcode" / "plugins"
    plugins_dir.mkdir(parents=True)
    (plugins_dir / "bad.py").write_text("raise RuntimeError('broken plugin')\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    registry = build_default_registry(tool_ctx)  # must not raise
    assert registry.get("read_file") is not None
