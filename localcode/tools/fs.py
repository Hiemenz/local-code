"""File tools: read, write, string-replace edit, unified-diff patch, ls, glob."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from localcode.errors import ToolError
from localcode.tools.base import Tool, ToolResult, truncate


def _numbered(text: str, start: int = 1) -> str:
    lines = text.splitlines()
    return "\n".join(f"{i + start:6d}\t{line}" for i, line in enumerate(lines))


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a file from the workspace. Returns content with line numbers. "
        "Use offset/limit for large files."
    )
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the project root"},
            "offset": {"type": "integer", "description": "1-based line to start from"},
            "limit": {"type": "integer", "description": "Max lines to return (default 2000)"},
        },
        "required": ["path"],
    }

    async def run(self, path: str, offset: int = 1, limit: int = 2000, **_: Any) -> ToolResult:
        target = self.ctx.resolve(path)
        if not target.is_file():
            return ToolResult.error(f"File not found: {path}")
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult.error(str(e))
        lines = text.splitlines()
        start = max(offset - 1, 0)
        window = lines[start : start + limit]
        body = _numbered("\n".join(window), start=start + 1)
        note = ""
        if start + limit < len(lines):
            note = f"\n... ({len(lines) - start - limit} more lines; re-read with offset={start + limit + 1})"
        return ToolResult(
            ok=True,
            output=truncate(body, self.ctx.max_output_chars()) + note,
            data={"path": str(target), "total_lines": len(lines)},
        )


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite a file with the given content. Creates parent directories."
    permission = "write"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    async def run(self, path: str, content: str, **_: Any) -> ToolResult:
        target = self.ctx.resolve(path)
        self.ctx.before_mutation(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(
            ok=True,
            output=f"Wrote {len(content)} chars to {path}",
            data={"path": str(target), "bytes": len(content.encode())},
        )

    def preview(self, args: dict[str, Any]) -> str:
        return f"write_file: {args.get('path')} ({len(str(args.get('content', '')))} chars)"


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. old_string must appear exactly once "
        "(include surrounding lines to disambiguate) unless replace_all is true."
    )
    permission = "write"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def run(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False, **_: Any
    ) -> ToolResult:
        target = self.ctx.resolve(path)
        if not target.is_file():
            return ToolResult.error(f"File not found: {path}")
        text = target.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            return ToolResult.error(
                f"old_string not found in {path}. Re-read the file; whitespace must match exactly."
            )
        if count > 1 and not replace_all:
            return ToolResult.error(
                f"old_string appears {count} times in {path}. Add context to make it "
                "unique, or set replace_all=true."
            )
        self.ctx.before_mutation(target)
        replaced = count if replace_all else 1
        new_text = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        target.write_text(new_text, encoding="utf-8")
        return ToolResult(
            ok=True,
            output=f"Replaced {replaced} occurrence(s) in {path}",
            data={"path": str(target), "replacements": replaced},
        )


class MultiEditTool(Tool):
    name = "multi_edit"
    description = (
        "Apply several exact string replacements to one file atomically: either "
        "all edits apply or none do. Edits run in order; each old_string must "
        "match the file as left by the previous edit."
    )
    permission = "write"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean", "default": False},
                    },
                    "required": ["old_string", "new_string"],
                },
                "minItems": 1,
            },
        },
        "required": ["path", "edits"],
    }

    async def run(self, path: str, edits: list[dict[str, Any]], **_: Any) -> ToolResult:
        target = self.ctx.resolve(path)
        if not target.is_file():
            return ToolResult.error(f"File not found: {path}")
        if not isinstance(edits, list) or not edits:
            return ToolResult.error("edits must be a non-empty array")
        text = target.read_text(encoding="utf-8")
        total = 0
        for i, edit in enumerate(edits, 1):
            old = str(edit.get("old_string", ""))
            new = str(edit.get("new_string", ""))
            replace_all = bool(edit.get("replace_all", False))
            count = text.count(old)
            if count == 0:
                return ToolResult.error(
                    f"Edit {i}/{len(edits)}: old_string not found; no edits were applied. "
                    "Remember each edit sees the result of the previous one."
                )
            if count > 1 and not replace_all:
                return ToolResult.error(
                    f"Edit {i}/{len(edits)}: old_string appears {count} times; no edits "
                    "were applied. Add context or set replace_all=true."
                )
            text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
            total += count if replace_all else 1
        self.ctx.before_mutation(target)
        target.write_text(text, encoding="utf-8")
        return ToolResult(
            ok=True,
            output=f"Applied {len(edits)} edit(s), {total} replacement(s) in {path}",
            data={"path": str(target), "edits": len(edits), "replacements": total},
        )


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_unified_diff(original: str, hunk_lines: list[list[str]]) -> str:
    """Apply parsed hunks to text. Each hunk is a list of ' ', '-', '+' lines."""
    src = original.splitlines()
    out: list[str] = []
    pos = 0
    for hunk in hunk_lines:
        old = [line[1:] for line in hunk if line[:1] in (" ", "-")]
        new = [line[1:] for line in hunk if line[:1] in (" ", "+")]
        if not old:  # pure addition (e.g. new file hunks)
            out.extend(src[pos:])
            src, pos = out + new, 0
            out = []
            continue
        idx = _find_block(src, old, pos)
        if idx < 0:
            raise ToolError(
                f"Hunk failed to apply; context not found:\n{'\n'.join(hunk[:5])}"
            )
        out.extend(src[pos:idx])
        out.extend(new)
        pos = idx + len(old)
    out.extend(src[pos:])
    trailing = "\n" if original.endswith("\n") or not original else ""
    return "\n".join(out) + trailing


def _find_block(haystack: list[str], needle: list[str], start: int) -> int:
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    # retry ignoring trailing whitespace
    stripped_needle = [line.rstrip() for line in needle]
    for i in range(start, len(haystack) - len(needle) + 1):
        if [line.rstrip() for line in haystack[i : i + len(needle)]] == stripped_needle:
            return i
    return -1


def parse_patch(patch: str) -> list[tuple[str, str, list[list[str]]]]:
    """Parse a unified diff into (old_path, new_path, hunks) per file."""
    files: list[tuple[str, str, list[list[str]]]] = []
    old_path = new_path = ""
    hunks: list[list[str]] = []
    current: list[str] | None = None

    def flush_file() -> None:
        nonlocal hunks, old_path, new_path
        if old_path or new_path:
            files.append((old_path, new_path, hunks))
        old_path = new_path = ""
        hunks = []

    for line in patch.splitlines():
        if line.startswith("--- "):
            flush_file()
            old_path = line[4:].strip().split("\t")[0]
            current = None
        elif line.startswith("+++ "):
            new_path = line[4:].strip().split("\t")[0]
        elif _HUNK_RE.match(line):
            current = []
            hunks.append(current)
        elif current is not None and line[:1] in (" ", "-", "+", ""):
            current.append(line if line else " ")
        elif line.startswith("\\ No newline"):
            continue
    flush_file()
    return files


def _strip_prefix(path: str) -> str:
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


class ApplyPatchTool(Tool):
    name = "apply_patch"
    description = (
        "Apply a unified diff (git diff format) to the workspace. Supports "
        "multi-file patches, file creation (--- /dev/null) and deletion (+++ /dev/null)."
    )
    permission = "write"
    parameters = {
        "type": "object",
        "properties": {"patch": {"type": "string", "description": "Unified diff text"}},
        "required": ["patch"],
    }

    async def run(self, patch: str, **_: Any) -> ToolResult:
        try:
            files = parse_patch(patch)
        except Exception as e:  # noqa: BLE001
            return ToolResult.error(f"Could not parse patch: {e}")
        if not files:
            return ToolResult.error("Patch contains no file sections")
        applied: list[str] = []
        for old_path, new_path, hunks in files:
            try:
                applied.append(self._apply_one(old_path, new_path, hunks))
            except (ToolError, OSError) as e:
                return ToolResult.error(
                    f"{e} (files already applied: {applied or 'none'})"
                )
        return ToolResult(ok=True, output="Applied patch to: " + ", ".join(applied))

    def _apply_one(self, old_path: str, new_path: str, hunks: list[list[str]]) -> str:
        if new_path == "/dev/null":  # deletion
            target = self.ctx.resolve(_strip_prefix(old_path))
            self.ctx.before_mutation(target)
            target.unlink(missing_ok=True)
            return f"{_strip_prefix(old_path)} (deleted)"
        rel = _strip_prefix(new_path)
        target = self.ctx.resolve(rel)
        if old_path == "/dev/null" or not target.exists():  # creation
            content = "\n".join(
                line[1:] for hunk in hunks for line in hunk if line[:1] == "+"
            )
            self.ctx.before_mutation(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content + "\n", encoding="utf-8")
            return f"{rel} (created)"
        original = target.read_text(encoding="utf-8")
        updated = apply_unified_diff(original, hunks)
        self.ctx.before_mutation(target)
        target.write_text(updated, encoding="utf-8")
        return rel


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories at a path (non-recursive)."
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "default": "."}},
    }

    async def run(self, path: str = ".", **_: Any) -> ToolResult:
        target = self.ctx.resolve(path)
        if not target.is_dir():
            return ToolResult.error(f"Not a directory: {path}")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        listing = "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries)
        return ToolResult(ok=True, output=listing or "(empty)")


class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern (e.g. 'src/**/*.py'), sorted by mtime."
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}},
        "required": ["pattern"],
    }

    async def run(self, pattern: str, **_: Any) -> ToolResult:
        root = self.ctx.root
        matches = [p for p in root.glob(pattern) if p.is_file() and not _ignored(p, root)]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        rels = [str(p.relative_to(root)) for p in matches[:200]]
        if not rels:
            return ToolResult(ok=True, output="No files matched.")
        more = f"\n... ({len(matches) - 200} more)" if len(matches) > 200 else ""
        return ToolResult(ok=True, output="\n".join(rels) + more, data={"count": len(matches)})


def _ignored(path: Path, root: Path) -> bool:
    rel = str(path.relative_to(root))
    for pat in (".git/*", "node_modules/*", "__pycache__/*", "*.pyc", ".venv/*", "venv/*"):
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, f"*/{pat}"):
            return True
    return False


TOOLS = [
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    MultiEditTool,
    ApplyPatchTool,
    ListDirTool,
    GlobTool,
]
