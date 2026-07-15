"""Streaming terminal rendering with rich: live markdown, tool lines, diffs."""

from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from localcode.agent.loop import AgentEvent, format_call


def _context_bar(used: int, total: int, width: int = 20) -> str:
    """Return a ▓░ progress bar string showing context token usage."""
    filled = min(round(used / total * width), width) if total else 0
    bar = "▓" * filled + "░" * (width - filled)
    k_used = used // 1000
    k_total = total // 1000
    return f"[{bar}] {k_used}k / {k_total}k ctx"


class StreamRenderer:
    """Accumulates streamed text and re-renders it as markdown, throttled
    so large responses don't overwhelm the terminal."""

    def __init__(
        self,
        console: Console,
        refresh_per_second: float = 12.0,
        context_length: int = 0,
    ) -> None:
        self.console = console
        self._buffer = ""
        self._live: Live | None = None
        self._last_refresh = 0.0
        self._interval = 1.0 / refresh_per_second
        self._context_length = context_length

    def feed(self, text: str) -> None:
        self._buffer += text
        if self._live is None:
            self._live = Live(
                Markdown(self._buffer), console=self.console,
                refresh_per_second=12, vertical_overflow="visible",
            )
            self._live.start()
        now = time.monotonic()
        if now - self._last_refresh >= self._interval:
            self._live.update(Markdown(self._buffer))
            self._last_refresh = now

    def flush(self) -> None:
        if self._live is not None:
            self._live.update(Markdown(self._buffer))
            self._live.stop()
            self._live = None
        self._buffer = ""

    def render_event(self, event: AgentEvent) -> None:
        if event.kind == "text":
            self.feed(event.text)
        elif event.kind == "tool_start" and event.tool_call is not None:
            self.flush()
            self.console.print(
                Text("  ⏺ ", style="green") + Text(format_call(event.tool_call), style="dim")
            )
        elif event.kind == "tool_end" and event.result is not None:
            first_line = event.result.output.split("\n", 1)[0][:120]
            style = "dim" if event.result.ok else "red"
            marker = "  ⎿ " if event.result.ok else "  ✗ "
            self.console.print(Text(marker + first_line, style=style))
        elif event.kind == "info":
            self.flush()
            self.console.print(Text(f"· {event.text}", style="italic dim"))
        elif event.kind == "turn_end":
            self.flush()
            ctx = event.stats.get("context_tokens", 0)
            if ctx and self._context_length:
                bar = _context_bar(int(ctx), self._context_length)
                self.console.print(Text(f"  {bar}", style="dim"))


def print_diff(console: Console, diff_text: str) -> None:
    if diff_text.strip():
        console.print(Syntax(diff_text, "diff", theme="ansi_dark", word_wrap=True))
    else:
        console.print("[dim]No changes.[/]")
