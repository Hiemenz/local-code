"""Execution tools: shell, Python, Node, tests, build."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from localcode.tools.base import Tool, ToolResult, truncate


async def run_subprocess(
    command: str, cwd: str, timeout: float, limit_chars: int
) -> ToolResult:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as e:
        return ToolResult.error(f"Failed to start command: {e}")
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return ToolResult.error(f"Command timed out after {timeout:.0f}s: {command}")
    text = stdout.decode(errors="replace")
    output = truncate(text, limit_chars)
    code = proc.returncode or 0
    if code != 0:
        output = f"[exit code {code}]\n{output}"
    return ToolResult(ok=code == 0, output=output or "(no output)", data={"exit_code": code})


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command in the project root and return combined stdout/stderr. "
        "Use for builds, installs, file operations, and anything without a dedicated tool."
    )
    permission = "execute"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number", "description": "Seconds (default 120, max 600)"},
        },
        "required": ["command"],
    }

    async def run(self, command: str, timeout: float = 120, **_: Any) -> ToolResult:
        return await run_subprocess(
            command, str(self.ctx.root), min(timeout, 600), self.ctx.max_output_chars()
        )

    def preview(self, args: dict[str, Any]) -> str:
        return f"$ {args.get('command', '')}"


class RunPythonTool(Tool):
    name = "run_python"
    description = "Execute a Python snippet in a subprocess and return its output."
    permission = "execute"
    parameters = {
        "type": "object",
        "properties": {"code": {"type": "string"}, "timeout": {"type": "number"}},
        "required": ["code"],
    }

    async def run(self, code: str, timeout: float = 60, **_: Any) -> ToolResult:
        cmd = f"{sys.executable} -c {_shquote(code)}"
        return await run_subprocess(
            cmd, str(self.ctx.root), min(timeout, 600), self.ctx.max_output_chars()
        )


class RunNodeTool(Tool):
    name = "run_node"
    description = "Execute a JavaScript snippet with Node.js and return its output."
    permission = "execute"
    parameters = {
        "type": "object",
        "properties": {"code": {"type": "string"}, "timeout": {"type": "number"}},
        "required": ["code"],
    }

    async def run(self, code: str, timeout: float = 60, **_: Any) -> ToolResult:
        cmd = f"node -e {_shquote(code)}"
        return await run_subprocess(
            cmd, str(self.ctx.root), min(timeout, 600), self.ctx.max_output_chars()
        )


class RunTestsTool(Tool):
    name = "run_tests"
    description = (
        "Run the project's test suite using the detected test framework. "
        "Optionally pass extra arguments (e.g. a test file or -k filter)."
    )
    permission = "execute"
    parameters = {
        "type": "object",
        "properties": {"args": {"type": "string", "description": "Extra CLI arguments"}},
    }

    async def run(self, args: str = "", **_: Any) -> ToolResult:
        command = self._test_command()
        if command is None:
            return ToolResult.error(
                "No test framework detected. Use the bash tool with an explicit command."
            )
        full = f"{command} {args}".strip()
        result = await run_subprocess(
            full, str(self.ctx.root), 600, self.ctx.max_output_chars()
        )
        result.output = f"$ {full}\n{result.output}"
        return result

    def _test_command(self) -> str | None:
        if self.ctx.index is not None and self.ctx.index.project.test_command:
            return self.ctx.index.project.test_command
        from localcode.index.detect import detect_project

        return detect_project(self.ctx.root).test_command


class RunBuildTool(Tool):
    name = "run_build"
    description = "Run the project's build command using the detected build system."
    permission = "execute"
    parameters = {"type": "object", "properties": {"args": {"type": "string"}}}

    async def run(self, args: str = "", **_: Any) -> ToolResult:
        from localcode.index.detect import detect_project

        info = (
            self.ctx.index.project
            if self.ctx.index is not None
            else detect_project(self.ctx.root)
        )
        if not info.build_command:
            return ToolResult.error("No build system detected.")
        full = f"{info.build_command} {args}".strip()
        result = await run_subprocess(full, str(self.ctx.root), 600, self.ctx.max_output_chars())
        result.output = f"$ {full}\n{result.output}"
        return result


def _shquote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


TOOLS = [BashTool, RunPythonTool, RunNodeTool, RunTestsTool, RunBuildTool]
