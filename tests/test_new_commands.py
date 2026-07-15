"""Tests for new slash commands: /commit, /pr, /doctor, /run, /search, and @file expansion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from localcode.cli.app import App, _expand_at_files
from localcode.cli.commands import dispatch
from localcode.config.schema import Config

# -- @file expansion -----------------------------------------------------------

def test_expand_at_files_existing(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("file content here", encoding="utf-8")
    result = _expand_at_files("Look at @hello.txt please", tmp_path)
    assert "file content here" in result
    assert "[File: hello.txt]" in result


def test_expand_at_files_missing_left_as_is(tmp_path: Path) -> None:
    result = _expand_at_files("no such @ghost.py here", tmp_path)
    assert result == "no such @ghost.py here"


def test_expand_at_files_no_at_unchanged(tmp_path: Path) -> None:
    result = _expand_at_files("plain text no expansion", tmp_path)
    assert result == "plain text no expansion"


def test_expand_at_files_multiple(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("aaa", encoding="utf-8")
    (tmp_path / "b.py").write_text("bbb", encoding="utf-8")
    result = _expand_at_files("@a.py and @b.py", tmp_path)
    assert "aaa" in result
    assert "bbb" in result


def test_expand_at_files_truncates_large(tmp_path: Path) -> None:
    big = "x" * 60_000
    (tmp_path / "big.txt").write_text(big, encoding="utf-8")
    result = _expand_at_files("@big.txt", tmp_path)
    assert "truncated" in result
    assert len(result) < 60_000 + 1000


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture
def app(
    workspace: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    fake_provider_factory: type,
) -> App:
    state = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_DATA_HOME", str(state / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(state / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(state / "config"))
    config = Config()
    config.agent.auto_test = False
    application = App(workspace, config)
    application.console = Console(record=True, width=120)
    fake = fake_provider_factory([("great commit message", []), ("PR description", [])])
    application.provider = fake
    application.agent.provider = fake
    application.agent.set_system_prompt("test prompt")
    return application


def output(app: App) -> str:
    return app.console.export_text()


# -- /commit -------------------------------------------------------------------

async def test_commit_nothing_staged(app: App, monkeypatch: pytest.MonkeyPatch) -> None:
    from localcode.tools import shell

    async def _fake_subprocess(cmd: str, cwd: str, timeout: float, limit: int) -> MagicMock:
        from localcode.tools.base import ToolResult
        return ToolResult(ok=True, output="(no output)")

    monkeypatch.setattr(shell, "run_subprocess", _fake_subprocess)
    await dispatch(app, "/commit")
    assert "Nothing staged" in output(app)


async def test_commit_stages_and_accepts(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localcode.tools import shell

    calls: list[str] = []

    async def _fake_subprocess(cmd: str, cwd: str, timeout: float, limit: int) -> MagicMock:
        from localcode.tools.base import ToolResult
        calls.append(cmd)
        if "stat" in cmd:
            return ToolResult(ok=True, output="1 file changed")
        if "diff --staged" in cmd and "stat" not in cmd:
            return ToolResult(ok=True, output="-old\n+new")
        if "commit" in cmd:
            return ToolResult(ok=True, output="[main abc1234] great commit message")
        return ToolResult(ok=True, output="ok")

    monkeypatch.setattr(shell, "run_subprocess", _fake_subprocess)
    monkeypatch.setattr(app.console, "input", lambda prompt: "y")
    await dispatch(app, "/commit")
    assert any("commit" in c for c in calls)


# -- /doctor -------------------------------------------------------------------

async def test_doctor_shows_table(app: App) -> None:
    await dispatch(app, "/doctor")
    text = output(app)
    assert "provider" in text
    assert "model" in text
    assert "tools" in text


# -- /search -------------------------------------------------------------------

async def test_search_no_sessions(app: App, monkeypatch: pytest.MonkeyPatch) -> None:
    from localcode.config import loader

    monkeypatch.setattr(loader, "data_dir", lambda: app.root / "_nonexistent_data")
    await dispatch(app, "/search anything")
    assert "No sessions" in output(app)


async def test_search_no_query(app: App) -> None:
    await dispatch(app, "/search")
    assert "Usage" in output(app)


async def test_search_finds_by_title(
    app: App, tmp_path: Path
) -> None:
    import time

    from localcode.session.store import SessionMeta

    sid = "20260101-000000-abc123"
    meta = SessionMeta(
        id=sid,
        created=time.time(),
        updated=time.time(),
        title="fix the login bug",
        project_root=str(app.root),
    )
    # Insert directly into the store's FTS index
    app.store._index.upsert(meta, "fix the login bug")

    await dispatch(app, "/search login")
    text = output(app)
    assert sid in text
    assert "fix the login bug" in text


# -- /run ----------------------------------------------------------------------

async def test_run_no_args(app: App) -> None:
    await dispatch(app, "/run")
    assert "Usage" in output(app)


async def test_run_starts_background_task(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localcode.config import loader

    monkeypatch.setattr(
        loader, "data_dir", lambda: app.root / "_bg_data"
    )
    tasks_created: list[object] = []

    def _mock_create_task(coro: object) -> MagicMock:
        tasks_created.append(coro)
        # Don't actually run; return a dummy task
        import asyncio
        return asyncio.ensure_future(coro)

    monkeypatch.setattr(__import__("asyncio"), "create_task", _mock_create_task)
    await dispatch(app, "/run hello world")
    assert "Background task started" in output(app)
