"""Search tools: ripgrep-backed grep with a pure-Python fallback."""

from __future__ import annotations

import asyncio
import fnmatch
import re
import shutil
from typing import Any

from localcode.tools.base import Tool, ToolResult, truncate

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".tox"}


def rg_available() -> bool:
    return shutil.which("rg") is not None


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents with a regular expression. Uses ripgrep when "
        "available. Returns matching lines as path:line:text."
    )
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {"type": "string", "description": "Directory or file to search", "default": "."},
            "glob": {"type": "string", "description": "Filter files, e.g. '*.py'"},
            "ignore_case": {"type": "boolean", "default": False},
            "max_results": {"type": "integer", "default": 100},
            "context": {"type": "integer", "default": 0,
                        "description": "Lines of context around each match"},
        },
        "required": ["pattern"],
    }

    async def run(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignore_case: bool = False,
        max_results: int = 100,
        context: int = 0,
        **_: Any,
    ) -> ToolResult:
        target = self.ctx.resolve(path)
        context = max(0, min(context, 10))
        if rg_available():
            lines = await self._rg(pattern, str(target), glob, ignore_case, context)
        else:
            try:
                lines = self._pygrep(pattern, target, glob, ignore_case, context)
            except re.error as e:
                return ToolResult.error(f"Invalid regex: {e}")
        total = len(lines)
        lines = lines[:max_results]
        # report paths relative to root for readability
        root_prefix = str(self.ctx.root.resolve()) + "/"
        body = "\n".join(line.removeprefix(root_prefix) for line in lines)
        if not body:
            return ToolResult(ok=True, output="No matches found.")
        more = f"\n... ({total - max_results} more matches)" if total > max_results else ""
        return ToolResult(
            ok=True,
            output=truncate(body, self.ctx.max_output_chars()) + more,
            data={"matches": total},
        )

    async def _rg(
        self, pattern: str, path: str, glob: str | None, ignore_case: bool, context: int = 0
    ) -> list[str]:
        cmd = ["rg", "--line-number", "--no-heading", "--color", "never", "-e", pattern]
        if ignore_case:
            cmd.append("-i")
        if context:
            cmd += ["-C", str(context)]
        if glob:
            cmd += ["--glob", glob]
        cmd.append(path)
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _stderr = await proc.communicate()
        return stdout.decode(errors="replace").splitlines()

    def _pygrep(
        self, pattern: str, target: Any, glob: str | None, ignore_case: bool, context: int = 0
    ) -> list[str]:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        results: list[str] = []
        files = [target] if target.is_file() else sorted(
            p
            for p in target.rglob("*")
            if p.is_file() and not (_SKIP_DIRS & set(part for part in p.parts))
        )
        for f in files:
            if glob and not fnmatch.fnmatch(f.name, glob):
                continue
            try:
                if f.stat().st_size > 2_000_000:
                    continue
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            for i, line in enumerate(lines, 1):
                if not rx.search(line):
                    continue
                if context:
                    lo = max(0, i - 1 - context)
                    hi = min(len(lines), i + context)
                    for j in range(lo, hi):
                        sep = ":" if j == i - 1 else "-"  # rg-style context marker
                        results.append(f"{f}{sep}{j + 1}{sep}{lines[j].strip()[:300]}")
                    results.append("--")
                else:
                    results.append(f"{f}:{i}:{line.strip()[:300]}")
                if len(results) >= 2000:
                    return results
        if results and results[-1] == "--":
            results.pop()
        return results


class FindFilesTool(Tool):
    name = "find_files"
    description = "Find files whose path contains a substring or matches a glob."
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Substring or glob"}},
        "required": ["query"],
    }

    async def run(self, query: str, **_: Any) -> ToolResult:
        root = self.ctx.root.resolve()
        is_glob = any(c in query for c in "*?[")
        hits: list[str] = []
        for p in root.rglob("*"):
            if not p.is_file() or _SKIP_DIRS & set(p.parts):
                continue
            rel = str(p.relative_to(root))
            if (fnmatch.fnmatch(rel, query) if is_glob else query.lower() in rel.lower()):
                hits.append(rel)
                if len(hits) >= 200:
                    break
        return ToolResult(ok=True, output="\n".join(hits) or "No files found.")


TOOLS = [GrepTool, FindFilesTool]
