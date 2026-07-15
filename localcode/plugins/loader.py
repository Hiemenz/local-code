"""Plugin loading. A plugin is any module that exposes one or both of:

    def register(registry: ToolRegistry, ctx: ToolContext) -> None
    def register_commands(register_fn: RegisterCommandFn) -> None

Plugins are discovered from (in order):
1. installed packages exposing the ``localcode.tools`` entry-point group,
2. module paths listed under ``plugins = [...]`` in config,
3. ``*.py`` files in ``~/.config/localcode/plugins/``.

``register_commands`` receives a callable with signature::

    register_fn(name: str, help_text: str, handler: Handler) -> None

where ``Handler = Callable[[App, str], Awaitable[bool]]``.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Awaitable, Callable
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING

from localcode.config.loader import user_config_path
from localcode.logging import get_logger

if TYPE_CHECKING:
    from localcode.cli.app import App
    from localcode.tools.base import ToolContext, ToolRegistry

log = get_logger("plugins")

# Type alias for the register_commands callback
RegisterCommandFn = Callable[[str, str, Callable[["App", str], Awaitable[bool]]], None]


def _call_register(register: object, registry: ToolRegistry, ctx: ToolContext, origin: str) -> None:
    try:
        register(registry, ctx)  # type: ignore[operator]
        log.info(f"loaded plugin tools: {origin}")
    except Exception as e:  # noqa: BLE001 - a broken plugin must not break startup
        log.warning(f"plugin {origin} failed to register tools: {e}")


def _call_register_commands(
    register_commands: object, register_fn: RegisterCommandFn, origin: str
) -> None:
    try:
        register_commands(register_fn)  # type: ignore[operator]
        log.info(f"loaded plugin commands: {origin}")
    except Exception as e:  # noqa: BLE001
        log.warning(f"plugin {origin} failed to register commands: {e}")


def load_plugins(registry: ToolRegistry, ctx: ToolContext) -> None:
    # Lazy import to avoid circular import (cli/commands imports cli/app under TYPE_CHECKING)
    from localcode.cli.commands import register_command

    # 1. entry points
    try:
        for ep in entry_points(group="localcode.tools"):
            try:
                mod = ep.load()
                if callable(getattr(mod, "register", None)):
                    _call_register(mod.register, registry, ctx, f"entry point {ep.name}")
                if callable(getattr(mod, "register_commands", None)):
                    _call_register_commands(
                        mod.register_commands, register_command, f"entry point {ep.name}"
                    )
            except Exception as e:  # noqa: BLE001
                log.warning(f"could not load entry point {ep.name}: {e}")
    except Exception:  # noqa: BLE001 - importlib.metadata quirks
        pass

    # 2. configured module paths
    for module_path in ctx.config.plugins:
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            log.warning(f"plugin module {module_path} not importable: {e}")
            continue
        fn = getattr(module, "register", None)
        if fn is None:
            log.warning(f"plugin module {module_path} has no register()")
        else:
            _call_register(fn, registry, ctx, module_path)
        cmd_fn = getattr(module, "register_commands", None)
        if cmd_fn is not None:
            _call_register_commands(cmd_fn, register_command, module_path)

    # 3. user plugins directory
    plugins_dir = user_config_path().parent / "plugins"
    if plugins_dir.is_dir():
        for path in sorted(plugins_dir.glob("*.py")):
            _load_file_plugin(path, registry, ctx, register_command)


def _load_file_plugin(
    path: Path,
    registry: ToolRegistry,
    ctx: ToolContext,
    register_fn: RegisterCommandFn,
) -> None:
    spec = importlib.util.spec_from_file_location(f"localcode_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:  # noqa: BLE001
        log.warning(f"plugin file {path} failed to import: {e}")
        return
    fn = getattr(module, "register", None)
    if fn is not None:
        _call_register(fn, registry, ctx, str(path))
    cmd_fn = getattr(module, "register_commands", None)
    if cmd_fn is not None:
        _call_register_commands(cmd_fn, register_fn, str(path))
