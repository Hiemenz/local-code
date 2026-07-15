"""Refactoring tools: word-boundary symbol rename across the repository."""

from __future__ import annotations

import re
from typing import Any

from localcode.tools.base import Tool, ToolResult


class RenameSymbolTool(Tool):
    name = "rename_symbol"
    description = (
        "Rename an identifier across the repository using word-boundary matching. "
        "Set dry_run=true first to preview which files and lines change."
    )
    permission = "write"
    parameters = {
        "type": "object",
        "properties": {
            "old_name": {"type": "string"},
            "new_name": {"type": "string"},
            "glob": {"type": "string", "description": "Limit to files matching, e.g. '*.py'"},
            "dry_run": {"type": "boolean", "default": False},
        },
        "required": ["old_name", "new_name"],
    }

    async def run(
        self,
        old_name: str,
        new_name: str,
        glob: str = "",
        dry_run: bool = False,
        **_: Any,
    ) -> ToolResult:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", old_name):
            return ToolResult.error("old_name must be a plain identifier")
        rx = re.compile(rf"\b{re.escape(old_name)}\b")
        root = self.ctx.root.resolve()

        if self.ctx.index is not None:
            candidates = [root / f for f in self.ctx.index.files]
        else:
            candidates = [p for p in root.rglob("*") if p.is_file()]

        import fnmatch

        changed: list[str] = []
        total = 0
        for path in candidates:
            if glob and not fnmatch.fnmatch(path.name, glob):
                continue
            if any(part in {".git", "node_modules", ".venv", "__pycache__"} for part in path.parts):
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            new_text, count = rx.subn(new_name, text)
            if count == 0:
                continue
            total += count
            rel = str(path.relative_to(root))
            changed.append(f"{rel}: {count} occurrence(s)")
            if not dry_run:
                self.ctx.before_mutation(path)
                path.write_text(new_text, encoding="utf-8")
        if not changed:
            return ToolResult(ok=True, output=f"No occurrences of '{old_name}' found.")
        action = "Would rename" if dry_run else "Renamed"
        body = f"{action} '{old_name}' -> '{new_name}' ({total} occurrences):\n" + "\n".join(changed)
        return ToolResult(ok=True, output=body, data={"files": len(changed), "occurrences": total})


TOOLS = [RenameSymbolTool]
