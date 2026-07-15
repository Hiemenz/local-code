"""Lifecycle hooks: run user-configured shell commands on agent events."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from localcode.config.schema import HookConfig
from localcode.logging import get_logger

log = get_logger("hooks")


class HookRunner:
    def __init__(self, hooks: list[HookConfig]) -> None:
        self._hooks = hooks

    async def fire(
        self,
        event: str,
        *,
        tool_name: str = "",
        extra_env: dict[str, str] | None = None,
    ) -> None:
        matched = [
            h for h in self._hooks
            if h.event == event
            and (not h.tool_filter or tool_name in h.tool_filter)
        ]
        if not matched:
            return
        env: dict[str, str] = {**os.environ}
        env["LOCALCODE_EVENT"] = event
        if tool_name:
            env["LOCALCODE_TOOL"] = tool_name
        if extra_env:
            env.update(extra_env)
        for hook in matched:
            await _run_hook(hook.command, env)


async def _run_hook(command: str, env: dict[str, str]) -> None:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            log.warning(f"hook timed out after 30s: {command!r}")
            return
        if proc.returncode != 0:
            log.warning(
                f"hook exited {proc.returncode}: {command!r} — "
                f"{stderr.decode(errors='replace')[:200]}"
            )
    except Exception as e:  # noqa: BLE001
        log.warning(f"hook failed: {command!r}: {e}")


def make_tool_env(args: dict[str, Any]) -> dict[str, str]:
    """Build extra env vars for pre_tool / post_tool events."""
    return {"LOCALCODE_TOOL_ARGS": json.dumps(args, ensure_ascii=False, default=str)[:2000]}


def make_result_env(ok: bool, output: str) -> dict[str, str]:
    return {
        "LOCALCODE_TOOL_OK": "1" if ok else "0",
        "LOCALCODE_TOOL_OUTPUT": output[:500],
    }
