"""Tests for QueryDbTool (SQLite)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from localcode.tools.base import ToolContext
from localcode.tools.dbtools import QueryDbTool


def _make_db(root: Path) -> Path:
    db = root / "test.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
    con.execute("INSERT INTO items VALUES (1, 'alpha', 9.5)")
    con.execute("INSERT INTO items VALUES (2, 'beta', 7.0)")
    con.commit()
    con.close()
    return db


async def test_query_select_rows(tool_ctx: ToolContext) -> None:
    _make_db(tool_ctx.root)
    result = await QueryDbTool(tool_ctx).run(path="test.db", sql="SELECT * FROM items")
    assert result.ok
    assert "alpha" in result.output
    assert "beta" in result.output
    assert result.data is not None
    assert result.data["row_count"] == 2
    assert "id" in result.data["columns"]


async def test_query_filtered(tool_ctx: ToolContext) -> None:
    _make_db(tool_ctx.root)
    result = await QueryDbTool(tool_ctx).run(
        path="test.db", sql="SELECT name FROM items WHERE score > 8"
    )
    assert result.ok
    assert "alpha" in result.output
    assert "beta" not in result.output


async def test_query_pragma(tool_ctx: ToolContext) -> None:
    _make_db(tool_ctx.root)
    result = await QueryDbTool(tool_ctx).run(path="test.db", sql="PRAGMA table_info(items)")
    assert result.ok
    assert "name" in result.output


async def test_query_blocks_insert(tool_ctx: ToolContext) -> None:
    _make_db(tool_ctx.root)
    result = await QueryDbTool(tool_ctx).run(
        path="test.db", sql="INSERT INTO items VALUES (3, 'x', 1.0)"
    )
    assert not result.ok
    assert "SELECT" in result.output or "PRAGMA" in result.output


async def test_query_blocks_drop(tool_ctx: ToolContext) -> None:
    _make_db(tool_ctx.root)
    result = await QueryDbTool(tool_ctx).run(path="test.db", sql="DROP TABLE items")
    assert not result.ok


async def test_query_missing_file(tool_ctx: ToolContext) -> None:
    result = await QueryDbTool(tool_ctx).run(path="nope.db", sql="SELECT 1")
    assert not result.ok
    assert "not found" in result.output.lower()


async def test_query_bad_sql(tool_ctx: ToolContext) -> None:
    _make_db(tool_ctx.root)
    result = await QueryDbTool(tool_ctx).run(path="test.db", sql="SELECT * FROM nonexistent")
    assert not result.ok


async def test_query_empty_result(tool_ctx: ToolContext) -> None:
    _make_db(tool_ctx.root)
    result = await QueryDbTool(tool_ctx).run(
        path="test.db", sql="SELECT * FROM items WHERE id = 999"
    )
    assert result.ok
    assert "0 rows" in result.output
