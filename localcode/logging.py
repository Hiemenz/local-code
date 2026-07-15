"""Structured logging setup.

Logs go to a rotating file in the user data directory as JSON lines;
console logging is off by default so it never corrupts the TUI.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "data", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_dir: Path, level: str = "INFO", console: bool = False) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("localcode")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "localcode.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(stream)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"localcode.{name}")
