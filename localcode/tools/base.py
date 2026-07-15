"""Tool abstraction: JSON-schema described, permissioned, async tools."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from localcode.errors import PathSecurityError
from localcode.providers.base import ToolSpec

if TYPE_CHECKING:
    from localcode.agent.checkpoints import CheckpointManager
    from localcode.config.schema import Config
    from localcode.index.repomap import RepoIndex

Permission = Literal["read", "write", "execute"]


@dataclass
class ToolResult:
    """Structured tool output. ``output`` is the text fed back to the model;
    ``data`` carries a machine-readable payload for the API and plugins."""

    ok: bool
    output: str
    data: dict[str, Any] | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {"ok": self.ok, "output": self.output}
        if self.data is not None:
            payload["data"] = self.data
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(ok=False, output=f"Error: {message}")


@dataclass
class ToolContext:
    """Shared state handed to every tool invocation."""

    root: Path
    config: Config
    checkpoints: CheckpointManager | None = None
    index: RepoIndex | None = None
    files_modified: set[Path] = field(default_factory=set)

    def resolve(self, path: str) -> Path:
        """Resolve a user/model-supplied path safely inside the workspace."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        root = self.root.resolve()
        if resolved != root and root not in resolved.parents:
            raise PathSecurityError(f"Path escapes workspace root: {path}")
        return resolved

    def before_mutation(self, path: Path) -> None:
        """Snapshot a file before changing it, and mark it modified."""
        if self.checkpoints is not None:
            self.checkpoints.snapshot(path)
        self.files_modified.add(path)

    def max_output_chars(self) -> int:
        return self.config.context.max_tool_result_tokens * 4


class Tool(ABC):
    name: str
    description: str
    permission: Permission = "read"
    parameters: dict[str, Any]  # JSON schema for arguments

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult: ...

    def preview(self, args: dict[str, Any]) -> str:
        """One-line human-readable description shown in approval prompts."""
        summary = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
        return f"{self.name}({summary})"


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def truncate(text: str, limit: int, note: str = "output truncated") -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{note}: {len(text)} chars total] ...\n{text[-half:]}"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def __iter__(self) -> Any:
        return iter(self._tools.values())


def build_default_registry(ctx: ToolContext) -> ToolRegistry:
    """Register all built-in tools, then plugin tools."""
    from localcode.plugins.loader import load_plugins
    from localcode.tools import (
        agent_tool,
        dbtools,
        fs,
        gittools,
        net,
        refactor,
        repo,
        search,
        shell,
    )

    registry = ToolRegistry()
    for module in (fs, search, shell, gittools, refactor, repo, net, dbtools, agent_tool):
        for tool_cls in module.TOOLS:
            registry.register(tool_cls(ctx))
    load_plugins(registry, ctx)
    return registry
