"""Units: logging, retries, argument parsing, api guard, small helpers."""

from __future__ import annotations

import json
import logging as stdlib_logging
from pathlib import Path

import pytest

from localcode.errors import LocalCodeError, ProviderError
from localcode.logging import get_logger, setup_logging
from localcode.providers.base import parse_arguments, with_retries
from localcode.tools.base import ToolResult, truncate

# -- logging ---------------------------------------------------------------

def read_log_lines(log_dir: Path) -> list[dict]:
    text = (log_dir / "localcode.log").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_logging_writes_json_lines(tmp_path: Path) -> None:
    setup_logging(tmp_path, level="INFO")
    log = get_logger("unit")
    log.info("hello world", extra={"data": {"key": "value"}})
    lines = read_log_lines(tmp_path)
    assert lines[-1]["msg"] == "hello world"
    assert lines[-1]["key"] == "value"
    assert lines[-1]["level"] == "INFO"
    assert lines[-1]["logger"] == "localcode.unit"


def test_logging_captures_exceptions(tmp_path: Path) -> None:
    setup_logging(tmp_path, level="INFO")
    log = get_logger("unit")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("it failed")
    lines = read_log_lines(tmp_path)
    assert "ValueError: boom" in lines[-1]["exc"]


def test_logging_level_filtering(tmp_path: Path) -> None:
    setup_logging(tmp_path, level="WARNING")
    get_logger("unit").info("should be dropped")
    get_logger("unit").warning("should appear")
    messages = [line["msg"] for line in read_log_lines(tmp_path)]
    assert "should be dropped" not in messages
    assert "should appear" in messages


def test_logging_console_handler_optional(tmp_path: Path) -> None:
    setup_logging(tmp_path, level="INFO", console=True)
    root = stdlib_logging.getLogger("localcode")
    kinds = {type(h).__name__ for h in root.handlers}
    assert "StreamHandler" in kinds
    setup_logging(tmp_path, level="INFO", console=False)  # handlers reset
    kinds = {type(h).__name__ for h in stdlib_logging.getLogger("localcode").handlers}
    assert "StreamHandler" not in kinds


# -- provider retry helper ------------------------------------------------------

async def test_with_retries_succeeds_after_transient_failures() -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderError("transient", retryable=True)
        return "ok"

    assert await with_retries(flaky, max_retries=3, base_delay=0.01) == "ok"
    assert attempts == 3


async def test_with_retries_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    async def always_fails() -> str:
        raise ProviderError("down", retryable=True)

    with pytest.raises(ProviderError):
        await with_retries(always_fails, max_retries=2, base_delay=0.01)


async def test_with_retries_non_retryable_immediate() -> None:
    attempts = 0

    async def fatal() -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderError("bad request", retryable=False)

    with pytest.raises(ProviderError):
        await with_retries(fatal, max_retries=5, base_delay=0.01)
    assert attempts == 1


# -- parse_arguments -------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, {}),
        ("", {}),
        ("  ", {}),
        ({"a": 1}, {"a": 1}),
        ('{"path": "x.py"}', {"path": "x.py"}),
        ("not json", {"_raw": "not json"}),
        ("[1, 2]", {"_raw": "[1, 2]"}),
    ],
)
def test_parse_arguments(raw, expected) -> None:
    assert parse_arguments(raw) == expected


# -- api guard --------------------------------------------------------------------

def test_api_requires_extras() -> None:
    pytest.importorskip("localcode.api.server")
    try:
        import fastapi  # noqa: F401

        pytest.skip("fastapi installed; guard not exercised")
    except ImportError:
        pass
    from localcode.api.server import create_app

    with pytest.raises(LocalCodeError, match="localcode\\[api\\]"):
        create_app(None, None)  # type: ignore[arg-type]


# -- helpers -----------------------------------------------------------------------

def test_truncate_middle_out() -> None:
    text = "A" * 600 + "MIDDLE" + "B" * 600
    out = truncate(text, 100)
    assert len(out) < len(text)
    assert out.startswith("A") and out.endswith("B")
    assert "truncated" in out
    assert truncate("short", 100) == "short"


def test_tool_result_json() -> None:
    result = ToolResult(ok=True, output="done", data={"n": 3})
    payload = json.loads(result.to_json())
    assert payload == {"ok": True, "output": "done", "data": {"n": 3}}
    assert json.loads(ToolResult.error("nope").to_json())["ok"] is False
