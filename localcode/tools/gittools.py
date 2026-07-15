"""Git tools: status, diff, commit, branch, log — structured wrappers over git."""

from __future__ import annotations

from typing import Any

from localcode.tools.base import Tool, ToolResult, truncate
from localcode.tools.shell import run_subprocess


class _GitTool(Tool):
    async def _git(self, args: str, timeout: float = 60) -> ToolResult:
        return await run_subprocess(
            f"git {args}", str(self.ctx.root), timeout, self.ctx.max_output_chars()
        )

    @staticmethod
    def _empty(result: ToolResult) -> bool:
        return result.output.strip() in ("", "(no output)")


class GitStatusTool(_GitTool):
    name = "git_status"
    description = "Show working tree status (short format) and current branch."
    permission = "read"
    parameters = {"type": "object", "properties": {}}

    async def run(self, **_: Any) -> ToolResult:
        result = await self._git("status --short --branch")
        if result.ok and self._empty(result):
            result.output = "Clean working tree."
        return result


class GitDiffTool(_GitTool):
    name = "git_diff"
    description = "Show changes as a unified diff. Set staged=true for the index; optionally limit to a path."
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {
            "staged": {"type": "boolean", "default": False},
            "path": {"type": "string"},
            "base": {"type": "string", "description": "Diff against this ref (e.g. main)"},
        },
    }

    async def run(
        self, staged: bool = False, path: str = "", base: str = "", **_: Any
    ) -> ToolResult:
        args = "diff"
        if staged:
            args += " --cached"
        if base:
            args += f" {base}"
        if path:
            args += f" -- {path}"
        result = await self._git(args)
        if result.ok and self._empty(result):
            result.output = "No changes."
        return result


class GitCommitTool(_GitTool):
    name = "git_commit"
    description = (
        "Commit changes. Stages the given paths (or all tracked changes when "
        "add_all=true) and commits with the message."
    )
    permission = "write"
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "add_all": {"type": "boolean", "default": False},
        },
        "required": ["message"],
    }

    async def run(
        self, message: str, paths: list[str] | None = None, add_all: bool = False, **_: Any
    ) -> ToolResult:
        if add_all:
            staged = await self._git("add -A")
            if not staged.ok:
                return staged
        elif paths:
            quoted = " ".join(f"'{p}'" for p in paths)
            staged = await self._git(f"add -- {quoted}")
            if not staged.ok:
                return staged
        msg = message.replace("'", "'\\''")
        return await self._git(f"commit -m '{msg}'")

    def preview(self, args: dict[str, Any]) -> str:
        return f"git commit -m {str(args.get('message', ''))[:60]!r}"


class GitBranchTool(_GitTool):
    name = "git_branch"
    description = "List branches, or create/switch to one."
    permission = "write"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "create", "switch"], "default": "list"},
            "name": {"type": "string"},
        },
    }

    async def run(self, action: str = "list", name: str = "", **_: Any) -> ToolResult:
        if action == "list":
            return await self._git("branch --all -v")
        if not name:
            return ToolResult.error("Branch name required.")
        safe = name.replace("'", "")
        if action == "create":
            return await self._git(f"checkout -b '{safe}'")
        if action == "switch":
            return await self._git(f"checkout '{safe}'")
        return ToolResult.error(f"Unknown action: {action}")


class GitLogTool(_GitTool):
    name = "git_log"
    description = "Show recent commit history."
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "default": 10},
            "path": {"type": "string"},
        },
    }

    async def run(self, count: int = 10, path: str = "", **_: Any) -> ToolResult:
        args = f"log --oneline --decorate -n {min(count, 100)}"
        if path:
            args += f" -- {path}"
        result = await self._git(args)
        result.output = truncate(result.output, self.ctx.max_output_chars())
        return result


TOOLS = [GitStatusTool, GitDiffTool, GitCommitTool, GitBranchTool, GitLogTool]
