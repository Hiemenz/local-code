
import pytest

from localcode.tools.base import ToolContext
from localcode.tools.gittools import (
    GitBranchTool,
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
)
from localcode.tools.shell import run_subprocess


@pytest.fixture
async def git_ctx(tool_ctx: ToolContext) -> ToolContext:
    root = str(tool_ctx.root)
    for cmd in (
        "git init -q -b main",
        "git config user.email t@t.t",
        "git config user.name t",
        "git add -A",
        "git commit -qm initial",
    ):
        result = await run_subprocess(cmd, root, 30, 10_000)
        assert result.ok, result.output
    return tool_ctx


async def test_status_clean(git_ctx: ToolContext) -> None:
    result = await GitStatusTool(git_ctx).run()
    assert result.ok
    assert "main" in result.output


async def test_diff_and_commit(git_ctx: ToolContext) -> None:
    (git_ctx.root / "src/util.py").write_text("def helper():\n    return 7\n")
    diff = await GitDiffTool(git_ctx).run()
    assert diff.ok and "return 7" in diff.output

    commit = await GitCommitTool(git_ctx).run(message="change helper", add_all=True)
    assert commit.ok, commit.output

    log = await GitLogTool(git_ctx).run(count=2)
    assert "change helper" in log.output

    diff_after = await GitDiffTool(git_ctx).run()
    assert "No changes" in diff_after.output


async def test_branch_create_and_list(git_ctx: ToolContext) -> None:
    created = await GitBranchTool(git_ctx).run(action="create", name="feature/x")
    assert created.ok, created.output
    listed = await GitBranchTool(git_ctx).run(action="list")
    assert "feature/x" in listed.output
    back = await GitBranchTool(git_ctx).run(action="switch", name="main")
    assert back.ok


async def test_commit_specific_paths(git_ctx: ToolContext) -> None:
    (git_ctx.root / "a.txt").write_text("a")
    (git_ctx.root / "b.txt").write_text("b")
    result = await GitCommitTool(git_ctx).run(message="only a", paths=["a.txt"])
    assert result.ok, result.output
    status = await GitStatusTool(git_ctx).run()
    assert "b.txt" in status.output  # b remains untracked
