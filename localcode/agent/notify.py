"""Desktop and push notifications for completed agent turns."""

from __future__ import annotations

import asyncio
import os

from localcode.logging import get_logger

log = get_logger("notify")


async def notify(title: str, body: str, backend: str = "auto", ntfy_topic: str = "") -> None:
    """Send a notification; silently no-ops when nothing is available."""
    if backend == "none":
        return
    if backend in ("auto", "notify-send"):
        if await _notify_send(title, body):
            return
        if backend == "notify-send":
            return
    if backend in ("auto", "ntfy"):
        topic = ntfy_topic or os.environ.get("LOCALCODE_NTFY_TOPIC", "")
        if topic:
            await _ntfy(title, body, topic)


async def _notify_send(title: str, body: str) -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "notify-send",
            "--expire-time=8000",
            title,
            body,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5.0)
        return proc.returncode == 0
    except (TimeoutError, FileNotFoundError, OSError):
        return False


async def _ntfy(title: str, body: str, topic: str) -> bool:
    try:
        import httpx  # already a project dependency

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"https://ntfy.sh/{topic}",
                content=body.encode(),
                headers={"Title": title},
            )
            return resp.status_code < 300
    except Exception:  # noqa: BLE001
        return False
