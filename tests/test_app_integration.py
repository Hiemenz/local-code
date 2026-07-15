"""Integration tests: the real App wired end-to-end against the scripted
mock OpenAI-compatible server from the benchmark suite. No model needed."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from mock_server import start as start_mock_server  # noqa: E402

from localcode.cli.app import App, _apply_env_overrides, build_parser  # noqa: E402
from localcode.config.schema import Config, ProviderConfig  # noqa: E402


@pytest.fixture(scope="module")
def mock_server() -> Iterator[int]:
    server, port = start_mock_server()
    yield port
    server.shutdown()


@pytest.fixture
def isolated_state(tmp_path_factory: pytest.TempPathFactory,
                   monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_DATA_HOME", str(state / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(state / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(state / "config"))
    return state


def make_config(port: int) -> Config:
    config = Config()
    config.provider = ProviderConfig(
        type="openai", base_url=f"http://127.0.0.1:{port}", model="mock-coder"
    )
    config.permissions.approval_mode = "full-auto"
    config.agent.auto_test = False
    return config


async def test_run_once_edits_file(
    mock_server: int, isolated_state: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "main.py").write_text("def greet():\n    return 'hi'\n")
    app = App(tmp_path, make_config(mock_server))
    exit_code = await app.run_once("change greet to return hello world")
    assert exit_code == 0
    assert "return 'hello world'" in (tmp_path / "main.py").read_text()
    out = capsys.readouterr()
    assert "hello world" in out.out          # final answer on stdout
    assert "read_file" in out.err            # tool activity on stderr
    # transcript persisted
    events = app.session.events()
    kinds = [e["type"] for e in events]
    assert kinds[0] == "user" and "tool_result" in kinds


async def test_run_once_checkpoints_are_undoable(
    mock_server: int, isolated_state: Path, tmp_path: Path
) -> None:
    (tmp_path / "main.py").write_text("def greet():\n    return 'hi'\n")
    app = App(tmp_path, make_config(mock_server))
    await app.run_once("edit please")
    assert "hello world" in (tmp_path / "main.py").read_text()
    restored = app.checkpoints.undo_last()
    assert restored
    assert (tmp_path / "main.py").read_text() == "def greet():\n    return 'hi'\n"


async def test_project_instructions_reach_system_prompt(
    mock_server: int, isolated_state: Path, tmp_path: Path
) -> None:
    (tmp_path / "main.py").write_text("def greet():\n    return 'hi'\n")
    (tmp_path / "CLAUDE.md").write_text("Never use tabs. Run make check before finishing.")
    app = App(tmp_path, make_config(mock_server))
    await app.startup()
    assert "Never use tabs" in app.agent._system_prompt
    assert "from CLAUDE.md" in app.agent._system_prompt
    await app.provider.close()


async def test_startup_builds_repo_map_into_prompt(
    mock_server: int, isolated_state: Path, tmp_path: Path
) -> None:
    (tmp_path / "core.py").write_text("def central():\n    pass\n")
    (tmp_path / "user.py").write_text("import core\ncore.central()\n")
    app = App(tmp_path, make_config(mock_server))
    await app.startup()
    prompt = app.agent._system_prompt
    assert "Repository map" in prompt and "core.py" in prompt
    await app.provider.close()


async def test_attach_session_restores_history(
    mock_server: int, isolated_state: Path, tmp_path: Path
) -> None:
    (tmp_path / "main.py").write_text("def greet():\n    return 'hi'\n")
    app = App(tmp_path, make_config(mock_server))
    await app.run_once("first turn")
    first_id = app.session.meta.id

    app2 = App(tmp_path, make_config(mock_server))
    resumed = app2.store.load(first_id)
    assert resumed is not None
    app2.attach_session(resumed)
    assert app2.agent.messages  # history restored
    assert app2.session.meta.id == first_id
    await app2.provider.close()


# -- argument parsing and env overrides ------------------------------------------

def test_build_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["/some/dir", "-p", "do it", "--mode", "full-auto",
                              "--model", "m1", "--resume"])
    assert args.path == "/some/dir"
    assert args.oneshot == "do it"
    assert args.mode == "full-auto"
    assert args.model == "m1"
    assert args.resume == "latest"


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.path == "."
    assert args.oneshot is None
    assert args.resume is None
    assert not args.cont


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALCODE_MODEL", "env-model")
    monkeypatch.setenv("LOCALCODE_BASE_URL", "http://env:1")
    monkeypatch.setenv("LOCALCODE_PROVIDER", "openai")
    config = Config()
    _apply_env_overrides(config)
    assert config.provider.model == "env-model"
    assert config.provider.base_url == "http://env:1"
    assert config.provider.type == "openai"
