import sys

from localcode.tools.base import ToolContext
from localcode.tools.refactor import RenameSymbolTool
from localcode.tools.search import FindFilesTool, GrepTool
from localcode.tools.shell import BashTool, RunPythonTool, RunTestsTool


async def test_grep_python_fallback(tool_ctx: ToolContext) -> None:
    result = await GrepTool(tool_ctx).run(pattern=r"def helper")
    assert result.ok
    assert "src/util.py" in result.output


async def test_grep_no_match(tool_ctx: ToolContext) -> None:
    result = await GrepTool(tool_ctx).run(pattern="zzz_nothing")
    assert result.ok
    assert "No matches" in result.output


async def test_grep_glob_filter(tool_ctx: ToolContext) -> None:
    result = await GrepTool(tool_ctx).run(pattern="Demo", glob="*.md")
    assert "README.md" in result.output


async def test_find_files(tool_ctx: ToolContext) -> None:
    result = await FindFilesTool(tool_ctx).run(query="util")
    assert "src/util.py" in result.output


async def test_bash_ok(tool_ctx: ToolContext) -> None:
    result = await BashTool(tool_ctx).run(command="echo hello")
    assert result.ok
    assert "hello" in result.output


async def test_bash_nonzero_exit(tool_ctx: ToolContext) -> None:
    result = await BashTool(tool_ctx).run(command="false")
    assert not result.ok
    assert result.data is not None and result.data["exit_code"] == 1


async def test_bash_timeout(tool_ctx: ToolContext) -> None:
    result = await BashTool(tool_ctx).run(command=f"{sys.executable} -c 'import time; time.sleep(5)'", timeout=0.4)
    assert not result.ok
    assert "timed out" in result.output


async def test_run_python(tool_ctx: ToolContext) -> None:
    result = await RunPythonTool(tool_ctx).run(code="print(2 + 2)")
    assert result.ok
    assert "4" in result.output


async def test_run_tests_no_framework(tool_ctx: ToolContext) -> None:
    result = await RunTestsTool(tool_ctx).run()
    assert not result.ok  # demo workspace has no test framework
    assert "No test framework" in result.output


async def test_rename_symbol_dry_run(tool_ctx: ToolContext) -> None:
    result = await RenameSymbolTool(tool_ctx).run(
        old_name="helper", new_name="assist", dry_run=True
    )
    assert result.ok
    assert "Would rename" in result.output
    assert "return util.helper()" in (tool_ctx.root / "src/app.py").read_text()


async def test_rename_symbol_applies(tool_ctx: ToolContext) -> None:
    result = await RenameSymbolTool(tool_ctx).run(old_name="helper", new_name="assist")
    assert result.ok
    assert result.data is not None and result.data["files"] == 2
    assert "util.assist()" in (tool_ctx.root / "src/app.py").read_text()
    assert "def assist" in (tool_ctx.root / "src/util.py").read_text()


async def test_rename_word_boundary(tool_ctx: ToolContext) -> None:
    (tool_ctx.root / "b.py").write_text("helper_extra = 1\nhelper = 2\n")
    await RenameSymbolTool(tool_ctx).run(old_name="helper", new_name="assist")
    text = (tool_ctx.root / "b.py").read_text()
    assert "helper_extra" in text and "assist = 2" in text
