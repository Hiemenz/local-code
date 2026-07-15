from pathlib import Path

from localcode.agent.prompts import build_system_prompt, load_project_instructions
from localcode.tools.base import ToolContext
from localcode.tools.fs import MultiEditTool
from localcode.tools.search import GrepTool

# -- project instruction files (CLAUDE.md workflow) ---------------------------

def test_load_instructions_precedence(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("agents rules")
    (tmp_path / "CLAUDE.md").write_text("claude rules")
    name, text = load_project_instructions(tmp_path)
    assert name == "CLAUDE.md" and text == "claude rules"
    (tmp_path / "LOCALCODE.md").write_text("localcode rules")
    name, text = load_project_instructions(tmp_path)
    assert name == "LOCALCODE.md"


def test_load_instructions_absent_or_empty(tmp_path: Path) -> None:
    assert load_project_instructions(tmp_path) is None
    (tmp_path / "CLAUDE.md").write_text("   \n")
    assert load_project_instructions(tmp_path) is None


def test_load_instructions_truncated(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("x" * 20_000)
    _, text = load_project_instructions(tmp_path)
    assert len(text) == 8000


def test_instructions_injected_into_prompt(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        tmp_path, repo_map=None, project_summary=None,
        instructions=("CLAUDE.md", "Always run make lint before committing."),
    )
    assert "Project instructions (from CLAUDE.md)" in prompt
    assert "make lint" in prompt


def test_prompt_without_instructions_unchanged(tmp_path: Path) -> None:
    prompt = build_system_prompt(tmp_path, None, None)
    assert "Project instructions" not in prompt


# -- multi_edit ----------------------------------------------------------------

async def test_multi_edit_applies_in_order(tool_ctx: ToolContext) -> None:
    result = await MultiEditTool(tool_ctx).run(
        path="src/util.py",
        edits=[
            {"old_string": "def helper():", "new_string": "def helper(x=0):"},
            {"old_string": "return 42", "new_string": "return 42 + x"},
        ],
    )
    assert result.ok, result.output
    text = (tool_ctx.root / "src/util.py").read_text()
    assert "def helper(x=0):" in text and "return 42 + x" in text
    assert result.data == {
        "path": str(tool_ctx.root / "src/util.py"), "edits": 2, "replacements": 2,
    }


async def test_multi_edit_is_atomic_on_failure(tool_ctx: ToolContext) -> None:
    original = (tool_ctx.root / "src/util.py").read_text()
    result = await MultiEditTool(tool_ctx).run(
        path="src/util.py",
        edits=[
            {"old_string": "return 42", "new_string": "return 99"},
            {"old_string": "DOES NOT EXIST", "new_string": "x"},
        ],
    )
    assert not result.ok
    assert "no edits were applied" in result.output
    assert (tool_ctx.root / "src/util.py").read_text() == original
    assert not tool_ctx.files_modified  # nothing snapshotted either


async def test_multi_edit_sees_previous_edit(tool_ctx: ToolContext) -> None:
    # second edit matches text produced by the first
    result = await MultiEditTool(tool_ctx).run(
        path="src/util.py",
        edits=[
            {"old_string": "return 42", "new_string": "return VALUE"},
            {"old_string": "return VALUE", "new_string": "return 7"},
        ],
    )
    assert result.ok
    assert "return 7" in (tool_ctx.root / "src/util.py").read_text()


async def test_multi_edit_ambiguity_rejected(tool_ctx: ToolContext) -> None:
    (tool_ctx.root / "dup.py").write_text("a = 1\na = 1\n")
    result = await MultiEditTool(tool_ctx).run(
        path="dup.py", edits=[{"old_string": "a = 1", "new_string": "b = 2"}]
    )
    assert not result.ok and "appears 2 times" in result.output


async def test_multi_edit_replace_all(tool_ctx: ToolContext) -> None:
    (tool_ctx.root / "dup.py").write_text("a = 1\na = 1\n")
    result = await MultiEditTool(tool_ctx).run(
        path="dup.py",
        edits=[{"old_string": "a = 1", "new_string": "b = 2", "replace_all": True}],
    )
    assert result.ok
    assert (tool_ctx.root / "dup.py").read_text() == "b = 2\nb = 2\n"


async def test_multi_edit_missing_file(tool_ctx: ToolContext) -> None:
    result = await MultiEditTool(tool_ctx).run(
        path="nope.py", edits=[{"old_string": "a", "new_string": "b"}]
    )
    assert not result.ok


# -- grep context lines --------------------------------------------------------

async def test_grep_context_lines(tool_ctx: ToolContext) -> None:
    (tool_ctx.root / "ctx.txt").write_text("one\ntwo\nTARGET\nfour\nfive\n")
    result = await GrepTool(tool_ctx).run(pattern="TARGET", context=1, glob="ctx.txt")
    assert result.ok
    assert "two" in result.output and "four" in result.output
    assert "one" not in result.output and "five" not in result.output
    # match line keeps ':' separator, context lines use '-'
    assert "ctx.txt:3:TARGET" in result.output
    assert "ctx.txt-2-two" in result.output


async def test_grep_context_clamped(tool_ctx: ToolContext) -> None:
    (tool_ctx.root / "ctx.txt").write_text("TARGET\n")
    result = await GrepTool(tool_ctx).run(pattern="TARGET", context=999, glob="ctx.txt")
    assert result.ok  # clamped to 10, no crash at file boundaries
