"""Checkpointing: snapshot files before each mutation so any turn can be undone."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from localcode.logging import get_logger

log = get_logger("checkpoints")


@dataclass
class Turn:
    id: int
    label: str
    ts: float
    # original absolute path -> snapshot filename, or None if file didn't exist
    files: dict[str, str | None] = field(default_factory=dict)


class CheckpointManager:
    def __init__(self, storage_dir: Path) -> None:
        self.dir = storage_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self.dir / "manifest.json"
        self.turns: list[Turn] = self._load()
        self._current: Turn | None = None

    def _load(self) -> list[Turn]:
        if not self._manifest.exists():
            return []
        try:
            data = json.loads(self._manifest.read_text(encoding="utf-8"))
            return [Turn(**t) for t in data]
        except (OSError, json.JSONDecodeError, TypeError):
            return []

    def _save(self) -> None:
        payload = [t.__dict__ for t in self.turns]
        self._manifest.write_text(json.dumps(payload), encoding="utf-8")

    def begin_turn(self, label: str) -> None:
        next_id = (self.turns[-1].id + 1) if self.turns else 1
        self._current = Turn(id=next_id, label=label[:120], ts=time.time())

    def end_turn(self) -> None:
        if self._current is not None and self._current.files:
            self.turns.append(self._current)
            self._save()
        self._current = None

    def snapshot(self, path: Path) -> None:
        """Record the pre-mutation state of a file (first touch per turn wins)."""
        if self._current is None:
            self.begin_turn("(implicit)")
        assert self._current is not None
        key = str(path.resolve())
        if key in self._current.files:
            return
        if path.exists():
            snap_name = f"t{self._current.id}_{len(self._current.files)}_{path.name}"
            shutil.copy2(path, self.dir / snap_name)
            self._current.files[key] = snap_name
        else:
            self._current.files[key] = None  # created this turn; undo deletes it

    def undo_last(self) -> list[str]:
        """Restore all files touched in the most recent turn. Returns paths."""
        if not self.turns:
            return []
        turn = self.turns.pop()
        restored: list[str] = []
        for original, snap_name in turn.files.items():
            target = Path(original)
            if snap_name is None:
                target.unlink(missing_ok=True)
                restored.append(f"{original} (deleted; was created in that turn)")
            else:
                snap = self.dir / snap_name
                if snap.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(snap, target)
                    restored.append(original)
        self._save()
        log.info(f"undid turn {turn.id}: {turn.label}", extra={"data": {"files": restored}})
        return restored

    def history(self) -> list[tuple[int, str, int]]:
        """(turn id, label, files touched) for display."""
        return [(t.id, t.label, len(t.files)) for t in self.turns]
