from pathlib import Path

from localcode.agent.checkpoints import CheckpointManager
from localcode.config.schema import ContextConfig
from localcode.context.manager import ContextManager, estimate_tokens
from localcode.providers.base import Message, ToolCall
from localcode.session.store import SessionStore


def _msgs(n: int) -> list[Message]:
    out: list[Message] = [Message(role="system", content="sys")]
    for i in range(n):
        out.append(Message(role="user", content=f"question {i} " + "x" * 200))
        out.append(Message(role="assistant", content=f"answer {i} " + "y" * 200))
    return out


def test_estimate_tokens() -> None:
    assert estimate_tokens("abcd" * 100) == 100


def test_needs_compaction_threshold() -> None:
    mgr = ContextManager(ContextConfig(compact_threshold=0.5), context_length=2000)
    assert not mgr.needs_compaction(_msgs(1))
    assert mgr.needs_compaction(_msgs(20))


async def test_compact_preserves_recent(fake_provider_factory) -> None:
    provider = fake_provider_factory([("SUMMARY OF EARLIER WORK", [])])
    mgr = ContextManager(ContextConfig(keep_recent_messages=4), context_length=1000)
    messages = _msgs(10)
    compacted = await mgr.compact(messages, provider)
    assert len(compacted) < len(messages)
    assert compacted[0].role == "system"
    assert "SUMMARY OF EARLIER WORK" in compacted[1].content
    assert compacted[-1].content == messages[-1].content
    # recent window starts at a user message, so no dangling tool results
    first_recent = compacted[2]
    assert first_recent.role == "user"


async def test_compact_never_splits_tool_pairs(fake_provider_factory) -> None:
    provider = fake_provider_factory([("S", [])])
    mgr = ContextManager(ContextConfig(keep_recent_messages=2), context_length=1000)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="a" * 500),
        Message(role="assistant", content="b" * 500),
        Message(role="user", content="do it"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="1", name="t", arguments={})]),
        Message(role="tool", content="result", tool_call_id="1", name="t"),
        Message(role="assistant", content="done"),
    ]
    compacted = await mgr.compact(messages, provider)
    for msg in compacted:
        if msg.role == "tool":
            ids = [c.id for m in compacted if m.role == "assistant" for c in m.tool_calls]
            assert msg.tool_call_id in ids


def test_checkpoint_undo_restores_and_deletes(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    existing = workdir / "a.txt"
    existing.write_text("original")
    mgr = CheckpointManager(tmp_path / "cp")

    mgr.begin_turn("edit a, create b")
    mgr.snapshot(existing)
    existing.write_text("modified")
    created = workdir / "b.txt"
    mgr.snapshot(created)  # doesn't exist yet
    created.write_text("new file")
    mgr.end_turn()

    restored = mgr.undo_last()
    assert existing.read_text() == "original"
    assert not created.exists()
    assert len(restored) == 2
    assert mgr.undo_last() == []  # nothing left


def test_checkpoint_persists_across_instances(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("v1")
    mgr = CheckpointManager(tmp_path / "cp")
    mgr.begin_turn("t")
    mgr.snapshot(target)
    target.write_text("v2")
    mgr.end_turn()

    mgr2 = CheckpointManager(tmp_path / "cp")
    assert mgr2.history()
    mgr2.undo_last()
    assert target.read_text() == "v1"


def test_session_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "data", tmp_path)
    session = store.create()
    session.append({"type": "user", "content": "fix the bug"})
    session.append(
        {
            "type": "assistant",
            "content": "ok",
            "tool_calls": [{"id": "1", "name": "read_file", "arguments": {"path": "a.py"}}],
        }
    )
    session.append({"type": "tool_result", "tool_call_id": "1", "name": "read_file",
                    "content": "data", "ok": True})
    session.append({"type": "assistant", "content": "fixed", "tool_calls": []})

    loaded = store.load(session.meta.id)
    assert loaded is not None
    messages = loaded.to_messages()
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1].tool_calls[0].name == "read_file"
    assert loaded.meta.title.startswith("fix the bug")


def test_session_list_and_latest(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "data", tmp_path)
    s1 = store.create()
    s1.append({"type": "user", "content": "first"})
    s2 = store.create()
    s2.append({"type": "user", "content": "second"})
    metas = store.list_sessions()
    assert len(metas) == 2
    latest = store.latest()
    assert latest is not None and latest.meta.id == s2.meta.id


def test_session_compaction_event_resets_history(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "data", tmp_path)
    session = store.create()
    session.append({"type": "user", "content": "old stuff"})
    session.append({"type": "compaction", "content": "[Conversation summary] ..."})
    session.append({"type": "assistant", "content": "continuing", "tool_calls": []})
    messages = session.to_messages()
    assert messages[0].content.startswith("[Conversation summary]")
    assert len(messages) == 2
