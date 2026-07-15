"""Rendering and approval-gate UI tests using a recording console."""

from __future__ import annotations

from rich.console import Console

from localcode.agent.loop import AgentEvent
from localcode.cli.approval import ConsoleApprover
from localcode.cli.render import StreamRenderer, print_diff
from localcode.providers.base import ToolCall
from localcode.tools.base import ToolContext, ToolResult
from localcode.tools.fs import EditFileTool, WriteFileTool
from localcode.tools.shell import BashTool


def make_console() -> Console:
    return Console(record=True, width=100, force_terminal=False)


# -- StreamRenderer ------------------------------------------------------------

def test_renderer_streams_and_flushes_markdown() -> None:
    console = make_console()
    renderer = StreamRenderer(console)
    renderer.render_event(AgentEvent(kind="text", text="Hello **wor"))
    renderer.render_event(AgentEvent(kind="text", text="ld**"))
    renderer.render_event(AgentEvent(kind="turn_end"))
    assert "Hello" in console.export_text()
    assert renderer._live is None  # flushed


def test_renderer_tool_lines() -> None:
    console = make_console()
    renderer = StreamRenderer(console)
    call = ToolCall(id="1", name="read_file", arguments={"path": "a.py"})
    renderer.render_event(AgentEvent(kind="tool_start", tool_call=call))
    renderer.render_event(
        AgentEvent(kind="tool_end", tool_call=call, result=ToolResult(ok=True, output="42 lines"))
    )
    renderer.render_event(
        AgentEvent(kind="tool_end", tool_call=call,
                   result=ToolResult(ok=False, output="Error: nope"))
    )
    text = console.export_text()
    assert "read_file" in text
    assert "42 lines" in text
    assert "Error: nope" in text


def test_renderer_info_and_interleaving() -> None:
    console = make_console()
    renderer = StreamRenderer(console)
    renderer.render_event(AgentEvent(kind="text", text="thinking..."))
    renderer.render_event(AgentEvent(kind="info", text="Tests passed."))
    text = console.export_text()
    assert "thinking..." in text and "Tests passed." in text


def test_print_diff_empty_and_content() -> None:
    console = make_console()
    print_diff(console, "")
    assert "No changes" in console.export_text()
    console2 = make_console()
    print_diff(console2, "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n")
    out = console2.export_text()
    assert "-old" in out and "+new" in out


# -- ConsoleApprover -----------------------------------------------------------

def make_approver(answers: list[str]) -> tuple[ConsoleApprover, Console]:
    console = make_console()
    replies = iter(answers)
    console.input = lambda prompt="": next(replies)  # type: ignore[method-assign]
    return ConsoleApprover(console), console


async def test_approver_decisions(tool_ctx: ToolContext) -> None:
    tool = BashTool(tool_ctx)
    for answer, expected in [("y", "allow"), ("a", "always"), ("n", "deny"), ("", "allow")]:
        approver, _ = make_approver([answer])
        assert await approver(tool, {"command": "ls"}) == expected


async def test_approver_reprompts_on_garbage(tool_ctx: ToolContext) -> None:
    approver, _ = make_approver(["maybe", "definitely", "n"])
    assert await approver(BashTool(tool_ctx), {"command": "ls"}) == "deny"


async def test_approver_shows_bash_command(tool_ctx: ToolContext) -> None:
    approver, console = make_approver(["y"])
    await approver(BashTool(tool_ctx), {"command": "rm -r build"})
    assert "rm -r build" in console.export_text()


async def test_approver_edit_shows_unified_diff(tool_ctx: ToolContext) -> None:
    approver, console = make_approver(["y"])
    await approver(
        EditFileTool(tool_ctx),
        {"path": "src/util.py", "old_string": "return 42", "new_string": "return 43"},
    )
    text = console.export_text()
    assert "-    return 42" in text
    assert "+    return 43" in text


async def test_approver_write_new_file_diff(tool_ctx: ToolContext) -> None:
    approver, console = make_approver(["y"])
    await approver(WriteFileTool(tool_ctx), {"path": "brand_new.py", "content": "x = 1\n"})
    assert "+x = 1" in console.export_text()


async def test_approver_diff_failure_falls_back(tool_ctx: ToolContext) -> None:
    # old_string not present -> no diff computable -> generic preview, no crash
    approver, console = make_approver(["y"])
    decision = await approver(
        EditFileTool(tool_ctx),
        {"path": "src/util.py", "old_string": "NOT THERE", "new_string": "x"},
    )
    assert decision == "allow"
    assert "edit_file" in console.export_text()
