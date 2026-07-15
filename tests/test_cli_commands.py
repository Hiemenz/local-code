"""Slash command tests against a real App instance (offline, FakeProvider)."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from localcode.cli.app import App
from localcode.cli.commands import COMMANDS, dispatch
from localcode.config.schema import Config


@pytest.fixture
def app(workspace: Path, tmp_path_factory: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch, fake_provider_factory) -> App:
    state = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_DATA_HOME", str(state / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(state / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(state / "config"))
    config = Config()
    config.agent.auto_test = False
    application = App(workspace, config)
    application.console = Console(record=True, width=100)
    fake = fake_provider_factory([("scripted reply", [])])
    application.provider = fake
    application.agent.provider = fake
    application.agent.set_system_prompt("test prompt")
    return application


def output(app: App) -> str:
    return app.console.export_text()


async def test_help_lists_every_command(app: App) -> None:
    assert await dispatch(app, "/help") is True
    text = output(app)
    for name in COMMANDS:
        assert f"/{name}" in text


async def test_unknown_command(app: App) -> None:
    assert await dispatch(app, "/frobnicate") is True
    assert "Unknown command" in output(app)


async def test_quit_and_exit_return_false(app: App) -> None:
    assert await dispatch(app, "/quit") is False
    assert await dispatch(app, "/exit") is False


async def test_tools_lists_registry_with_permissions(app: App) -> None:
    await dispatch(app, "/tools")
    text = output(app)
    for name in ("read_file", "multi_edit", "bash", "git_commit"):
        assert name in text
    assert "execute" in text and "write" in text and "read" in text


async def test_clear_resets_messages(app: App) -> None:
    from localcode.providers.base import Message

    app.agent.messages.append(Message(role="user", content="hi"))
    await dispatch(app, "/clear")
    assert app.agent.messages == []


async def test_mode_set_and_invalid(app: App) -> None:
    await dispatch(app, "/mode auto-edit")
    assert app.agent.mode == "auto-edit"
    await dispatch(app, "/mode yolo")
    assert app.agent.mode == "auto-edit"  # unchanged
    assert "Modes:" in output(app)


async def test_plan_toggle(app: App) -> None:
    await dispatch(app, "/plan")
    assert app.agent.mode == "plan"
    assert "Plan mode" in app.agent._system_prompt  # prompt rebuilt with plan note
    await dispatch(app, "/plan")
    assert app.agent.mode == "ask"


async def test_cost_reports_usage(app: App) -> None:
    app.agent.total_usage.prompt_tokens = 123
    await dispatch(app, "/cost")
    text = output(app)
    assert "123" in text and "current context" in text


async def test_model_show_and_switch(app: App) -> None:
    await dispatch(app, "/model")
    assert app.config.provider.model in output(app)
    await dispatch(app, "/model other-model")
    assert app.config.provider.model == "other-model"


async def test_model_switch_named_profile(app: App) -> None:
    from localcode.config.schema import ProviderConfig

    app.config.providers["fast"] = ProviderConfig(type="ollama", model="tiny")
    await dispatch(app, "/model fast")
    assert app.config.provider.model == "tiny"
    await app.provider.close()


async def test_sessions_resume_roundtrip(app: App) -> None:
    app.session.append({"type": "user", "content": "remember me"})
    old_id = app.session.meta.id
    await dispatch(app, "/sessions")
    assert old_id in output(app)
    await dispatch(app, f"/resume {old_id}")
    assert app.session.meta.id == old_id
    assert any(m.content == "remember me" for m in app.agent.messages)


async def test_resume_unknown_session(app: App) -> None:
    await dispatch(app, "/resume nope-123")
    assert "not found" in output(app)


async def test_undo_nothing(app: App) -> None:
    await dispatch(app, "/undo")
    assert "Nothing to undo" in output(app)


async def test_undo_after_checkpointed_turn(app: App) -> None:
    target = app.root / "src" / "util.py"
    original = target.read_text()
    app.checkpoints.begin_turn("edit util")
    app.checkpoints.snapshot(target)
    target.write_text("clobbered")
    app.checkpoints.end_turn()
    await dispatch(app, "/history")
    assert "edit util" in output(app)
    await dispatch(app, "/undo")
    assert target.read_text() == original


async def test_map_and_init(app: App) -> None:
    await dispatch(app, "/init")
    assert "Indexed" in output(app)
    await dispatch(app, "/map")
    assert "src/util.py" in output(app)


async def test_config_masks_api_key(app: App) -> None:
    app.config.provider.api_key = "secret-key"
    await dispatch(app, "/config")
    text = output(app)
    assert "secret-key" not in text
    assert "***" in text


async def test_compact_empty(app: App) -> None:
    await dispatch(app, "/compact")
    assert "Nothing to compact" in output(app)
