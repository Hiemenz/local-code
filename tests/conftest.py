from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localcode.agent.checkpoints import CheckpointManager
from localcode.config.schema import Config, ProviderConfig
from localcode.providers.base import Message, Provider, StreamEvent, ToolCall, ToolSpec, Usage
from localcode.tools.base import ToolContext


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "import util\n\n\ndef main():\n    return util.helper()\n", encoding="utf-8"
    )
    (tmp_path / "src" / "util.py").write_text(
        "def helper():\n    return 42\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def tool_ctx(workspace: Path, config: Config, tmp_path: Path) -> ToolContext:
    checkpoints = CheckpointManager(tmp_path / "_checkpoints")
    return ToolContext(root=workspace, config=config, checkpoints=checkpoints)


class FakeProvider(Provider):
    """Scripted provider: pops (text, tool_calls) responses in order."""

    def __init__(self, script: list[tuple[str, list[ToolCall]]]) -> None:
        super().__init__(ProviderConfig())
        self.script = list(script)
        self.requests: list[list[Message]] = []

    async def _chat_native(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append(list(messages))
        if not self.script:
            yield StreamEvent(kind="text", text="(script exhausted)")
            return
        text, calls = self.script.pop(0)
        # stream text in two chunks to exercise incremental paths
        if text:
            mid = len(text) // 2
            for part in (text[:mid], text[mid:]):
                if part:
                    yield StreamEvent(kind="text", text=part)
        for call in calls:
            yield StreamEvent(kind="tool_call", tool_call=call)
        yield StreamEvent(kind="usage", usage=Usage(prompt_tokens=10, completion_tokens=5))

    async def list_models(self) -> list[str]:
        return ["fake"]

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_provider_factory():
    return FakeProvider
