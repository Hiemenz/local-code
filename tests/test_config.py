from pathlib import Path

import pytest

from localcode.config.loader import _deep_merge, load_config
from localcode.config.schema import Config
from localcode.errors import ConfigError


def test_defaults() -> None:
    cfg = Config()
    assert cfg.provider.type == "ollama"
    assert cfg.permissions.approval_mode == "ask"
    assert cfg.agent.auto_test is True


def test_deep_merge() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 9}, "e": 4}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"b": 9, "c": 2}, "d": 3, "e": 4}


def test_project_config_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    (tmp_path / ".localcode.toml").write_text(
        '[provider]\nmodel = "deepseek-coder-v2:16b"\ntemperature = 0.7\n'
        '[permissions]\napproval_mode = "auto-edit"\n',
        encoding="utf-8",
    )
    cfg = load_config(project_root=tmp_path)
    assert cfg.provider.model == "deepseek-coder-v2:16b"
    assert cfg.provider.temperature == 0.7
    assert cfg.permissions.approval_mode == "auto-edit"


def test_explicit_config_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    (tmp_path / ".localcode.toml").write_text('[provider]\nmodel = "a"\n', encoding="utf-8")
    explicit = tmp_path / "override.toml"
    explicit.write_text('[provider]\nmodel = "b"\n', encoding="utf-8")
    cfg = load_config(project_root=tmp_path, explicit=explicit)
    assert cfg.provider.model == "b"


def test_invalid_toml_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    (tmp_path / ".localcode.toml").write_text("not [valid", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(project_root=tmp_path)


def test_invalid_value_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    (tmp_path / ".localcode.toml").write_text(
        '[permissions]\napproval_mode = "yolo"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_config(project_root=tmp_path)


def test_named_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    (tmp_path / ".localcode.toml").write_text(
        '[providers.fast]\ntype = "openai"\nbase_url = "http://localhost:1234"\n'
        'model = "qwen2.5-coder-7b"\n',
        encoding="utf-8",
    )
    cfg = load_config(project_root=tmp_path)
    assert cfg.providers["fast"].type == "openai"
