"""Configuration schema. Every user-facing behavior is configurable here."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ApprovalMode = Literal["plan", "ask", "auto-edit", "full-auto"]
ToolMode = Literal["auto", "native", "json"]
HookEvent = Literal["pre_tool", "post_tool", "session_start", "session_end", "turn_end"]


class HookConfig(BaseModel):
    """A single lifecycle hook: run ``command`` in a shell when ``event`` fires."""

    event: HookEvent
    command: str
    # Empty list = fire for every tool; otherwise only when tool name is in this list.
    tool_filter: list[str] = Field(default_factory=list)


class NotificationConfig(BaseModel):
    enabled: bool = False
    backend: Literal["auto", "notify-send", "ntfy", "none"] = "auto"
    # ntfy topic; also read from $LOCALCODE_NTFY_TOPIC env var.
    ntfy_topic: str = ""


class ProviderConfig(BaseModel):
    """Connection settings for one LLM backend."""

    type: Literal["ollama", "openai", "anthropic"] = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: str = ""  # unused by local servers; kept for OpenAI-compatible ones
    model: str = "qwen2.5-coder:14b"
    temperature: float = 0.2
    max_tokens: int = 4096
    context_length: int = 32768
    timeout_seconds: float = 300.0
    # native = use the server's tool-calling API; json = parse <tool_call> blocks
    # from plain text (works with any model); auto = native, fall back to json.
    tool_mode: ToolMode = "auto"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    # passed through to the backend (e.g. Ollama num_ctx / top_p)
    options: dict[str, float | int | str | bool] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    max_tool_rounds: int = 40
    max_retries: int = 3
    retry_base_delay: float = 1.0
    auto_test: bool = True  # run detected test command after edits
    fix_iterations: int = 3  # attempts to fix failing tests before returning
    verify_edits: bool = True  # re-read files after editing to confirm changes
    parallel_tools: bool = True  # run read-only tool calls concurrently


class ContextConfig(BaseModel):
    compact_threshold: float = 0.8  # fraction of context_length that triggers compaction
    keep_recent_messages: int = 8  # never summarized
    repo_map_tokens: int = 2048  # budget for the repository map in the system prompt
    max_tool_result_tokens: int = 4000  # tool output beyond this is truncated


class IndexConfig(BaseModel):
    enabled: bool = True
    max_file_size: int = 1_000_000
    ignore: list[str] = Field(
        default_factory=lambda: ["node_modules/", ".venv/", "venv/", "dist/", "build/",
                                "__pycache__/", ".git/", "*.min.js", "*.lock"]
    )
    use_tree_sitter: bool = True  # fall back to regex parsing when unavailable


class PermissionsConfig(BaseModel):
    approval_mode: ApprovalMode = "ask"
    # shell command prefixes that never require approval, e.g. ["git status", "ls"]
    allow_commands: list[str] = Field(default_factory=lambda: [
        "git status", "git diff", "git log", "ls", "cat", "pwd", "which", "grep", "find",
    ])
    # tool names that never require approval regardless of mode
    allow_tools: list[str] = Field(default_factory=list)
    deny_commands: list[str] = Field(default_factory=lambda: [
        "rm -rf /", "sudo rm", "mkfs", "shutdown", "reboot",
    ])


class CacheConfig(BaseModel):
    enabled: bool = True
    directory: Path | None = None  # default: ~/.cache/localcode
    prompt_cache: bool = True  # keep stable prefix ordering so server KV-cache hits
    index_ttl_seconds: int = 0  # 0 = mtime-based invalidation only


class LoggingConfig(BaseModel):
    level: str = "INFO"
    console: bool = False
    directory: Path | None = None  # default: ~/.local/share/localcode/logs


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class Config(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    # named alternative providers selectable with /model or [providers.<name>]
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    plugins: list[str] = Field(default_factory=list)  # extra plugin module paths
    hooks: list[HookConfig] = Field(default_factory=list)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
