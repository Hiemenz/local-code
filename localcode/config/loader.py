"""Layered TOML configuration loading.

Precedence (later wins): built-in defaults < ~/.config/localcode/config.toml
< <project>/.localcode.toml < explicit --config file.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from localcode.config.schema import Config
from localcode.errors import ConfigError


def user_config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "localcode" / "config.toml"


def data_dir() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "localcode"


def cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "localcode"


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(project_root: Path | None = None, explicit: Path | None = None) -> Config:
    """Load and merge all configuration layers."""
    data: dict[str, Any] = {}
    layers: list[Path] = [user_config_path()]
    if project_root is not None:
        layers.append(project_root / ".localcode.toml")
    if explicit is not None:
        if not explicit.exists():
            raise ConfigError(f"Config file not found: {explicit}")
        layers.append(explicit)
    for path in layers:
        if path.exists():
            data = _deep_merge(data, _read_toml(path))
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration: {e}") from e
