"""Session persistence: append-only JSONL transcripts + SQLite FTS index."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from localcode.providers.base import Message, ToolCall


@dataclass
class SessionMeta:
    id: str
    created: float
    updated: float
    title: str
    project_root: str


class SessionIndex:
    """SQLite FTS5 index over session metadata for instant /search across all projects."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._init_db()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(str(self._path))
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _init_db(self) -> None:
        self._has_fts5 = False
        try:
            with self._connect() as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS sessions_meta (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT '',
                        project_root TEXT NOT NULL DEFAULT '',
                        created REAL NOT NULL DEFAULT 0,
                        updated REAL NOT NULL DEFAULT 0,
                        first_message TEXT NOT NULL DEFAULT ''
                    )
                """)
                # Standalone FTS5 table — stores its own content
                try:
                    con.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
                            id UNINDEXED,
                            title,
                            project_root,
                            first_message,
                            tokenize='unicode61'
                        )
                    """)
                    self._has_fts5 = True
                except sqlite3.OperationalError:
                    pass
        except Exception:  # noqa: BLE001
            pass

    def upsert(self, meta: SessionMeta, first_message: str = "") -> None:
        try:
            with self._connect() as con:
                con.execute(
                    "INSERT OR REPLACE INTO sessions_meta "
                    "(id, title, project_root, created, updated, first_message) "
                    "VALUES (?,?,?,?,?,?)",
                    (meta.id, meta.title, meta.project_root,
                     meta.created, meta.updated, first_message[:1000]),
                )
                if self._has_fts5:
                    con.execute("DELETE FROM sessions_fts WHERE id = ?", (meta.id,))
                    con.execute(
                        "INSERT INTO sessions_fts(id, title, project_root, first_message) "
                        "VALUES (?,?,?,?)",
                        (meta.id, meta.title, meta.project_root, first_message[:1000]),
                    )
        except Exception:  # noqa: BLE001
            pass

    def search(self, query: str, limit: int = 20) -> list[tuple[str, str, str]]:
        """Return (id, title, project_root) tuples matching query."""
        try:
            with self._connect() as con:
                if self._has_fts5:
                    safe = query.replace('"', '""')
                    try:
                        rows = con.execute(
                            "SELECT id, title, project_root FROM sessions_fts "
                            f'WHERE sessions_fts MATCH \'"{safe}"\' '
                            "ORDER BY rank LIMIT ?",
                            (limit,),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        # Bad FTS5 query syntax — fall through to LIKE
                        rows = []
                    if rows:
                        return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]
                like = f"%{query}%"
                rows = con.execute(
                    "SELECT id, title, project_root FROM sessions_meta "
                    "WHERE title LIKE ? OR project_root LIKE ? OR first_message LIKE ? "
                    "ORDER BY updated DESC LIMIT ?",
                    (like, like, like, limit),
                ).fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]

    def rebuild_from_meta_files(self, sessions_root: Path) -> None:
        """Populate the index from existing .meta.json files (one-time migration)."""
        for meta_path in sessions_root.rglob("*.meta.json"):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                meta = SessionMeta(**data)
                jsonl = meta_path.with_suffix("").with_suffix(".jsonl")
                first_msg = ""
                if jsonl.exists():
                    for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
                        try:
                            ev = json.loads(line)
                            if ev.get("type") == "user":
                                first_msg = str(ev.get("content", ""))[:500]
                                break
                        except json.JSONDecodeError:
                            continue
                self.upsert(meta, first_msg)
            except Exception:  # noqa: BLE001
                continue


class Session:
    def __init__(
        self,
        path: Path,
        meta: SessionMeta,
        index: SessionIndex | None = None,
    ) -> None:
        self.path = path
        self.meta = meta
        self._index = index
        self._first_message: str = ""

    def append(self, event: dict[str, object]) -> None:
        event.setdefault("ts", round(time.time(), 3))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self.meta.updated = time.time()
        is_first_user = (
            event.get("type") == "user" and self.meta.title == "(new session)"
        )
        if is_first_user:
            self.meta.title = str(event.get("content", ""))[:80] or "(untitled)"
            self._first_message = str(event.get("content", ""))[:500]
        self._save_meta()
        if self._index is not None and is_first_user:
            self._index.upsert(self.meta, self._first_message)

    def _save_meta(self) -> None:
        meta_path = self.path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(self.meta.__dict__), encoding="utf-8")

    def events(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        out: list[dict[str, object]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def to_messages(self) -> list[Message]:
        """Rebuild the provider message history for resuming."""
        messages: list[Message] = []
        for event in self.events():
            kind = event.get("type")
            if kind == "user":
                messages.append(Message(role="user", content=str(event.get("content", ""))))
            elif kind == "assistant":
                raw_calls = event.get("tool_calls", [])
                if not isinstance(raw_calls, list):
                    raw_calls = []
                calls = [
                    ToolCall(
                        id=str(c.get("id", "")),
                        name=str(c.get("name", "")),
                        arguments=dict(c.get("arguments") or {}),
                    )
                    for c in raw_calls
                    if isinstance(c, dict)
                ]
                messages.append(
                    Message(
                        role="assistant",
                        content=str(event.get("content", "")),
                        tool_calls=calls,
                    )
                )
            elif kind == "tool_result":
                messages.append(
                    Message(
                        role="tool",
                        content=str(event.get("content", "")),
                        tool_call_id=str(event.get("tool_call_id", "")),
                        name=str(event.get("name", "")),
                    )
                )
            elif kind == "compaction":
                messages = [Message(role="user", content=str(event.get("content", "")))]
        return messages


class SessionStore:
    def __init__(self, base_dir: Path, project_root: Path) -> None:
        digest = hashlib.sha1(str(project_root.resolve()).encode()).hexdigest()[:16]
        self.dir = base_dir / "sessions" / digest
        self.dir.mkdir(parents=True, exist_ok=True)
        self.project_root = project_root
        self._index = SessionIndex(base_dir / "sessions" / "index.db")
        # Populate the index from disk on first run
        self._maybe_seed_index(base_dir / "sessions")

    def _maybe_seed_index(self, sessions_root: Path) -> None:
        """Rebuild the FTS index if it appears empty (first run after upgrade)."""
        try:
            with self._index._connect() as con:
                count = con.execute(
                    "SELECT COUNT(*) FROM sessions_meta"
                ).fetchone()[0]
            if count == 0:
                self._index.rebuild_from_meta_files(sessions_root)
        except Exception:  # noqa: BLE001
            pass

    def create(self) -> Session:
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        meta = SessionMeta(
            id=sid,
            created=time.time(),
            updated=time.time(),
            title="(new session)",
            project_root=str(self.project_root),
        )
        session = Session(self.dir / f"{sid}.jsonl", meta, index=self._index)
        session._save_meta()
        return session

    def list_sessions(self) -> list[SessionMeta]:
        metas: list[SessionMeta] = []
        for meta_path in self.dir.glob("*.meta.json"):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                metas.append(SessionMeta(**data))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        metas.sort(key=lambda m: -m.updated)
        return metas

    def load(self, session_id: str) -> Session | None:
        path = self.dir / f"{session_id}.jsonl"
        meta_path = self.dir / f"{session_id}.meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = SessionMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return Session(path, meta, index=self._index)

    def latest(self) -> Session | None:
        metas = self.list_sessions()
        return self.load(metas[0].id) if metas else None

    def search(self, query: str, limit: int = 20) -> list[tuple[str, str, str]]:
        """Full-text search across all sessions. Returns (id, title, project_root)."""
        return self._index.search(query, limit)
