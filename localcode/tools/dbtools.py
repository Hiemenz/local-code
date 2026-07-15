"""Database tools: query SQLite databases with read-only SQL."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from localcode.tools.base import Tool, ToolResult, truncate


def _run_query(db_path: Path, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        cur = con.cursor()
        cur.execute(sql)
        cols = [desc[0] for desc in (cur.description or [])]
        rows = cur.fetchmany(500)
        return cols, rows
    finally:
        con.close()


class QueryDbTool(Tool):
    name = "query_db"
    description = (
        "Execute a read-only SQL query against a SQLite database file in the workspace "
        "and return up to 500 rows as a formatted table. "
        "Only SELECT and PRAGMA statements are permitted."
    )
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative path to the .db file"},
            "sql": {"type": "string", "description": "SQL SELECT or PRAGMA query to run"},
        },
        "required": ["path", "sql"],
    }

    async def run(self, path: str, sql: str, **_: Any) -> ToolResult:
        upper = sql.strip().upper()
        if not (upper.startswith("SELECT") or upper.startswith("PRAGMA")):
            return ToolResult.error("Only SELECT and PRAGMA statements are permitted.")
        try:
            db_path = self.ctx.resolve(path)
        except Exception as e:  # noqa: BLE001
            return ToolResult.error(str(e))
        if not db_path.is_file():
            return ToolResult.error(f"File not found: {path}")
        try:
            cols, rows = await asyncio.to_thread(_run_query, db_path, sql)
        except sqlite3.Error as e:
            return ToolResult.error(str(e))
        if not cols:
            return ToolResult(ok=True, output=f"Query returned {len(rows)} row(s).")
        widths = [
            max(len(c), max((len(str(r[i])) for r in rows), default=0))
            for i, c in enumerate(cols)
        ]
        header = "  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True))
        sep = "  ".join("-" * w for w in widths)
        body_lines = [
            "  ".join(str(r[i]).ljust(w) for i, w in enumerate(widths)) for r in rows
        ]
        text = "\n".join([header, sep, *body_lines]) if rows else f"{header}\n{sep}\n(0 rows)"
        return ToolResult(
            ok=True,
            output=truncate(text, self.ctx.max_output_chars()),
            data={"columns": cols, "row_count": len(rows)},
        )


TOOLS = [QueryDbTool]
