"""Interactive approval gate: y / a(lways) / n, with diff previews for edits."""

from __future__ import annotations

import asyncio
import difflib
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from localcode.agent.loop import Decision
from localcode.tools.base import Tool


class ConsoleApprover:
    def __init__(self, console: Console) -> None:
        self.console = console

    async def __call__(self, tool: Tool, args: dict[str, Any]) -> Decision:
        self._show_preview(tool, args)
        while True:
            answer = (
                await asyncio.to_thread(
                    self.console.input,
                    r"[bold yellow]Allow?[/] [green]\[y]es[/] / [cyan]\[a]lways[/] / [red]\[n]o[/]: ",
                )
            ).strip().lower()
            if answer in ("y", "yes", ""):
                return "allow"
            if answer in ("a", "always"):
                return "always"
            if answer in ("n", "no"):
                return "deny"

    def _show_preview(self, tool: Tool, args: dict[str, Any]) -> None:
        title = f"[bold]{tool.name}[/] wants to run"
        diff = self._diff_for(tool, args)
        if diff is not None:
            self.console.print(Panel(Syntax(diff, "diff", theme="ansi_dark"), title=title))
        elif tool.name == "bash":
            self.console.print(
                Panel(Syntax(str(args.get("command", "")), "bash", theme="ansi_dark"), title=title)
            )
        else:
            self.console.print(Panel(tool.preview(args), title=title))

    def _diff_for(self, tool: Tool, args: dict[str, Any]) -> str | None:
        """Unified diff preview for file-mutating tools."""
        try:
            if tool.name == "edit_file":
                target = tool.ctx.resolve(str(args["path"]))
                if not target.is_file():
                    return None
                old_text = target.read_text(encoding="utf-8")
                old_string, new_string = str(args["old_string"]), str(args["new_string"])
                if old_string not in old_text:
                    return None
                new_text = (
                    old_text.replace(old_string, new_string)
                    if args.get("replace_all")
                    else old_text.replace(old_string, new_string, 1)
                )
            elif tool.name == "write_file":
                target = tool.ctx.resolve(str(args["path"]))
                old_text = target.read_text(encoding="utf-8") if target.is_file() else ""
                new_text = str(args["content"])
            elif tool.name == "apply_patch":
                return str(args.get("patch", ""))[:4000]
            else:
                return None
        except Exception:  # noqa: BLE001 - preview must never block approval
            return None
        rel = str(args.get("path", ""))
        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(), new_text.splitlines(),
                fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="",
            )
        )
        text = "\n".join(diff_lines)
        return text[:4000] if text else None
