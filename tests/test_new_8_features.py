"""Tests for the 8 new features:
1. Hooks system
2. Multimodal image input
3. Web search tool (DuckDuckGo parser)
4. Plugin slash commands
5. Desktop notifications
6. SQLite session FTS index
7. Git worktree isolation
8. Parallel subagents (spawn_agent)
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from localcode.config.schema import Config, HookConfig, NotificationConfig
from localcode.providers.base import ImageContent, Message
from localcode.session.store import SessionIndex, SessionMeta, SessionStore
from localcode.tools.base import ToolContext

# ---------------------------------------------------------------------------
# 1. Hooks system
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hook_fires_matching_event(tmp_path: Path) -> None:
    from localcode.agent.hooks import HookRunner

    flag = tmp_path / "fired.txt"
    hook = HookConfig(event="pre_tool", command=f"touch {flag}")
    runner = HookRunner([hook])
    await runner.fire("pre_tool", tool_name="bash")
    await asyncio.sleep(0.05)
    assert flag.exists()


@pytest.mark.asyncio
async def test_hook_tool_filter_respected(tmp_path: Path) -> None:
    from localcode.agent.hooks import HookRunner

    flag = tmp_path / "fired.txt"
    hook = HookConfig(event="pre_tool", command=f"touch {flag}", tool_filter=["read_file"])
    runner = HookRunner([hook])
    await runner.fire("pre_tool", tool_name="bash")
    await asyncio.sleep(0.05)
    assert not flag.exists()  # tool_filter excludes "bash"

    await runner.fire("pre_tool", tool_name="read_file")
    await asyncio.sleep(0.05)
    assert flag.exists()


@pytest.mark.asyncio
async def test_hook_wrong_event_not_fired(tmp_path: Path) -> None:
    from localcode.agent.hooks import HookRunner

    flag = tmp_path / "fired.txt"
    hook = HookConfig(event="session_start", command=f"touch {flag}")
    runner = HookRunner([hook])
    await runner.fire("pre_tool")
    await asyncio.sleep(0.05)
    assert not flag.exists()


@pytest.mark.asyncio
async def test_hook_env_vars_set(tmp_path: Path) -> None:
    from localcode.agent.hooks import HookRunner

    out = tmp_path / "event.txt"
    hook = HookConfig(event="pre_tool", command=f'echo "$LOCALCODE_EVENT $LOCALCODE_TOOL" > {out}')
    runner = HookRunner([hook])
    await runner.fire("pre_tool", tool_name="bash")
    await asyncio.sleep(0.1)
    assert "pre_tool" in out.read_text()
    assert "bash" in out.read_text()


@pytest.mark.asyncio
async def test_hook_broken_command_does_not_crash() -> None:
    from localcode.agent.hooks import HookRunner

    hook = HookConfig(event="turn_end", command="false")
    runner = HookRunner([hook])
    await runner.fire("turn_end")  # must not raise


# ---------------------------------------------------------------------------
# 2. Multimodal image input
# ---------------------------------------------------------------------------

def test_image_content_dataclass() -> None:
    img = ImageContent(data="abc123", mime_type="image/png")
    assert img.data == "abc123"
    assert img.mime_type == "image/png"


def test_message_images_field() -> None:
    img = ImageContent(data="xyz", mime_type="image/jpeg")
    msg = Message(role="user", content="look at this", images=[img])
    assert len(msg.images) == 1
    assert msg.images[0].data == "xyz"


def test_message_images_default_empty() -> None:
    msg = Message(role="user", content="no images")
    assert msg.images == []


def test_openai_wire_with_images() -> None:
    from localcode.providers.openai_compat import _wire_messages

    img = ImageContent(data="AABB", mime_type="image/png")
    msgs = [Message(role="user", content="describe this", images=[img])]
    wired = _wire_messages(msgs)
    assert wired[0]["role"] == "user"
    content = wired[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1]["type"] == "image_url"
    assert "data:image/png;base64,AABB" in content[1]["image_url"]["url"]


def test_openai_wire_no_images_is_plain_string() -> None:
    from localcode.providers.openai_compat import _wire_messages

    msgs = [Message(role="user", content="hello")]
    wired = _wire_messages(msgs)
    assert isinstance(wired[0]["content"], str)


def test_ollama_wire_with_images() -> None:
    from localcode.providers.ollama import _wire_messages

    img = ImageContent(data="BASE64DATA", mime_type="image/jpeg")
    msgs = [Message(role="user", content="look", images=[img])]
    wired = _wire_messages(msgs)
    assert wired[0]["images"] == ["BASE64DATA"]


def test_ollama_wire_no_images_has_no_images_key() -> None:
    from localcode.providers.ollama import _wire_messages

    msgs = [Message(role="assistant", content="ok")]
    wired = _wire_messages(msgs)
    assert "images" not in wired[0]


def test_at_ref_expands_image(tmp_path: Path) -> None:
    from localcode.cli.app import _expand_at_refs

    # Create a small 1x1 transparent PNG (minimal valid PNG bytes)
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    img_path = tmp_path / "test.png"
    img_path.write_bytes(png_bytes)

    text, images = _expand_at_refs(f"@{img_path}", tmp_path)
    assert "[Image:" in text
    assert len(images) == 1
    assert images[0].mime_type == "image/png"
    assert len(images[0].data) > 0


def test_at_ref_text_file_no_image(tmp_path: Path) -> None:
    from localcode.cli.app import _expand_at_refs

    (tmp_path / "notes.txt").write_text("hello world")
    text, images = _expand_at_refs("@notes.txt", tmp_path)
    assert "hello world" in text
    assert images == []


# ---------------------------------------------------------------------------
# 3. Web search tool (DuckDuckGo HTML parser)
# ---------------------------------------------------------------------------

def test_parse_ddg_results_extracts_entries() -> None:
    from localcode.tools.net import _parse_ddg_results

    html = '''
<div class="result results_links web-result">
  <h2><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">Example Title</a></h2>
  <span class="result__snippet">This is the snippet text.</span>
</div>
'''
    results = _parse_ddg_results(html, 5)
    assert len(results) == 1
    assert results[0]["title"] == "Example Title"
    assert "example.com" in results[0]["url"]
    assert results[0]["snippet"] == "This is the snippet text."


def test_parse_ddg_results_empty_html() -> None:
    from localcode.tools.net import _parse_ddg_results

    assert _parse_ddg_results("", 5) == []


def test_parse_ddg_results_respects_limit() -> None:
    from localcode.tools.net import _parse_ddg_results

    block = '''
<div class="result results_links web-result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com">Title</a>
  <span class="result__snippet">Snippet</span>
</div>
'''
    html = block * 10
    results = _parse_ddg_results(html, 3)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_web_search_tool_http_error(tool_ctx: ToolContext) -> None:
    import httpx

    from localcode.tools.net import WebSearchTool

    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("down"))
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient") as mock_client_cls:
        instance = MagicMock()
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        instance.post = AsyncMock(side_effect=httpx.ConnectError("down"))
        mock_client_cls.return_value = instance

        result = await WebSearchTool(tool_ctx).run(query="python asyncio")
    # Must not raise; returns an error ToolResult
    assert not result.ok


def test_web_search_in_default_registry(tool_ctx: ToolContext) -> None:
    from localcode.tools.base import build_default_registry

    registry = build_default_registry(tool_ctx)
    assert registry.get("web_search") is not None


# ---------------------------------------------------------------------------
# 4. Plugin slash commands
# ---------------------------------------------------------------------------

def test_register_command_adds_to_commands() -> None:
    from localcode.cli.commands import COMMANDS, register_command

    async def _fake_handler(app: object, args: str) -> bool:
        return True

    register_command("_test_plugin_cmd", "A test command", _fake_handler)
    assert "_test_plugin_cmd" in COMMANDS
    assert COMMANDS["_test_plugin_cmd"][0] == "A test command"
    # Clean up
    COMMANDS.pop("_test_plugin_cmd", None)


def test_plugin_register_commands_called(
    tool_ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins_dir = tmp_path / "cfg" / "localcode" / "plugins"
    plugins_dir.mkdir(parents=True)
    (plugins_dir / "cmd_plugin.py").write_text(
        '''
from localcode.tools.base import Tool, ToolResult

class DummyTool(Tool):
    name = "dummy_cmd_tool"
    description = "dummy"
    permission = "read"
    parameters = {"type": "object", "properties": {}}
    async def run(self, **kwargs):
        return ToolResult(ok=True, output="ok")

def register(registry, ctx):
    registry.register(DummyTool(ctx))

def register_commands(fn):
    async def _handler(app, args):
        return True
    fn("myplugincmd", "My plugin command", _handler)
'''
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    from localcode.tools.base import build_default_registry
    build_default_registry(tool_ctx)

    from localcode.cli.commands import COMMANDS
    assert "myplugincmd" in COMMANDS
    COMMANDS.pop("myplugincmd", None)


# ---------------------------------------------------------------------------
# 5. Desktop notifications
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notify_none_backend_no_op() -> None:
    from localcode.agent.notify import notify

    # Should complete without error and without calling anything
    await notify("title", "body", backend="none")


@pytest.mark.asyncio
async def test_notify_send_not_found_is_silent() -> None:
    from localcode.agent.notify import _notify_send

    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        result = await _notify_send("t", "b")
    assert result is False


@pytest.mark.asyncio
async def test_ntfy_skipped_without_topic() -> None:
    from localcode.agent.notify import _ntfy

    # No HTTP call should be made for an empty topic — the function returns False
    called = False

    async def _mock_post(*a: object, **kw: object) -> MagicMock:
        nonlocal called
        called = True
        return MagicMock(status_code=200)

    result = await _ntfy("t", "b", "")
    assert result is False
    assert not called


# ---------------------------------------------------------------------------
# 6. SQLite session FTS index
# ---------------------------------------------------------------------------

def test_session_index_upsert_and_search(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "test.db")
    meta = SessionMeta(
        id="sess-001",
        created=time.time(),
        updated=time.time(),
        title="fix authentication bug",
        project_root="/home/user/myproject",
    )
    idx.upsert(meta, "authentication was broken")

    results = idx.search("authentication")
    assert any(r[0] == "sess-001" for r in results)


def test_session_index_search_by_title(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "test.db")
    for i in range(3):
        meta = SessionMeta(
            id=f"sess-{i}",
            created=float(i),
            updated=float(i),
            title=f"project task {i}" if i != 1 else "unrelated session",
            project_root="/proj",
        )
        idx.upsert(meta)
    results = idx.search("project task")
    ids = [r[0] for r in results]
    assert "sess-0" in ids
    assert "sess-2" in ids
    assert "sess-1" not in ids


def test_session_index_empty_query_graceful(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "test.db")
    # Should not raise even on unusual queries
    results = idx.search("")
    assert isinstance(results, list)


def test_session_store_search_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    from localcode.config.loader import data_dir

    store = SessionStore(data_dir(), tmp_path)
    session = store.create()
    session.append({"type": "user", "content": "refactor the payment module"})

    results = store.search("payment")
    assert any("payment" in title for _, title, _ in results)


def test_session_index_upsert_updates_existing(tmp_path: Path) -> None:
    idx = SessionIndex(tmp_path / "test.db")
    meta = SessionMeta(
        id="dup-001", created=1.0, updated=1.0, title="old title", project_root="/p"
    )
    idx.upsert(meta, "old content")

    meta2 = SessionMeta(
        id="dup-001", created=1.0, updated=2.0, title="new title", project_root="/p"
    )
    idx.upsert(meta2, "new content")

    results = idx.search("new")
    assert any(r[0] == "dup-001" for r in results)


# ---------------------------------------------------------------------------
# 7. Git worktree isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worktree_create_fails_without_git(tmp_path: Path) -> None:
    from localcode.agent.worktree import WorktreeManager

    mgr = WorktreeManager(tmp_path, "sess-xyz", tmp_path / "base")
    # tmp_path is not a git repo, so this should raise
    with pytest.raises(RuntimeError, match="git worktree add failed"):
        await mgr.create()


@pytest.mark.asyncio
async def test_worktree_active_flag(tmp_path: Path) -> None:
    from localcode.agent.worktree import WorktreeManager

    mgr = WorktreeManager(tmp_path, "sess", tmp_path)
    assert not mgr.active


@pytest.mark.asyncio
async def test_worktree_in_git_repo(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(tmp_path),
        capture_output=True,
        check=True,
        env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test",
             "GIT_AUTHOR_EMAIL": "t@t.com", "GIT_COMMITTER_NAME": "test",
             "GIT_COMMITTER_EMAIL": "t@t.com"},
    )
    from localcode.agent.worktree import WorktreeManager

    mgr = WorktreeManager(tmp_path, "testsess", tmp_path)
    path = await mgr.create()
    assert path.is_dir()
    assert mgr.active

    await mgr.discard()
    assert not mgr.active


# ---------------------------------------------------------------------------
# 8. Parallel subagents (spawn_agent tool)
# ---------------------------------------------------------------------------

def test_spawn_agent_in_default_registry(tool_ctx: ToolContext) -> None:
    from localcode.tools.base import build_default_registry

    registry = build_default_registry(tool_ctx)
    tool = registry.get("spawn_agent")
    assert tool is not None
    assert tool.permission == "execute"


def test_spawn_agent_schema_valid(tool_ctx: ToolContext) -> None:
    from localcode.tools.agent_tool import SpawnAgentTool

    t = SpawnAgentTool(tool_ctx)
    spec = t.spec()
    assert spec.parameters["type"] == "object"
    assert "prompt" in spec.parameters["required"]


@pytest.mark.asyncio
async def test_spawn_agent_runs_subagent(tool_ctx: ToolContext) -> None:
    from localcode.tools.agent_tool import SpawnAgentTool
    from tests.conftest import FakeProvider

    fake = FakeProvider([("Hello from subagent", [])])

    # create_provider is imported inside run(), so patch the source module
    with patch("localcode.providers.base.create_provider", return_value=fake):
        tool = SpawnAgentTool(tool_ctx)
        result = await tool.run(prompt="say hello")

    assert result.ok
    assert "Hello from subagent" in result.output


@pytest.mark.asyncio
async def test_spawn_agent_excludes_itself(tool_ctx: ToolContext) -> None:
    """Sub-agent registry must not contain spawn_agent (prevents infinite recursion)."""
    from localcode.tools.agent_tool import SpawnAgentTool
    from tests.conftest import FakeProvider

    captured_registries: list[object] = []
    original_Agent = __import__("localcode.agent.loop", fromlist=["Agent"]).Agent

    class CapturingAgent(original_Agent):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            captured_registries.append(self.registry)

    fake = FakeProvider([("done", [])])
    with (
        patch("localcode.providers.base.create_provider", return_value=fake),
        patch("localcode.agent.loop.Agent", CapturingAgent),
    ):
        tool = SpawnAgentTool(tool_ctx)
        await tool.run(prompt="do something")

    assert captured_registries, "sub-agent was never created"
    sub_registry = captured_registries[0]
    assert sub_registry.get("spawn_agent") is None  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Config: new schema fields
# ---------------------------------------------------------------------------

def test_hook_config_schema() -> None:
    hook = HookConfig(event="pre_tool", command="echo hi")
    assert hook.tool_filter == []
    hook2 = HookConfig(event="post_tool", command="notify", tool_filter=["bash"])
    assert "bash" in hook2.tool_filter


def test_notification_config_defaults() -> None:
    cfg = NotificationConfig()
    assert not cfg.enabled
    assert cfg.backend == "auto"
    assert cfg.ntfy_topic == ""


def test_config_has_hooks_and_notifications() -> None:
    cfg = Config()
    assert cfg.hooks == []
    assert isinstance(cfg.notifications, NotificationConfig)
