"""Git worktree isolation: run the agent in a detached worktree.

Usage:
    mgr = WorktreeManager(repo_root, session_id, base_dir)
    worktree_path = await mgr.create()
    # ... agent runs with tool_ctx.root = worktree_path ...
    result = await mgr.merge_back()   # or await mgr.discard()
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from localcode.logging import get_logger

log = get_logger("worktree")


class WorktreeManager:
    def __init__(self, repo_root: Path, session_id: str, base_dir: Path) -> None:
        self._repo = repo_root
        self._session_id = session_id
        self._path = base_dir / "worktrees" / session_id
        self._active = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def active(self) -> bool:
        return self._active

    async def create(self) -> Path:
        """Create the git worktree and return its path."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "add", "--detach", str(self._path),
            cwd=str(self._repo),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"git worktree add failed: {stderr.decode(errors='replace')}"
            )
        self._active = True
        log.info(f"worktree created at {self._path}")
        return self._path

    async def merge_back(self) -> str:
        """Stage, commit everything in the worktree, and cherry-pick to the parent branch."""
        for cmd in (
            ["git", "add", "-A"],
            ["git", "commit", "--allow-empty", "--allow-empty-message",
             "-m", f"localcode session {self._session_id}"],
        ):
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(self._path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=str(self._path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        commit_hash = stdout.decode().strip()

        proc = await asyncio.create_subprocess_exec(
            "git", "cherry-pick", commit_hash,
            cwd=str(self._repo),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace")[:400]
            return (
                f"Cherry-pick failed: {err}\n"
                f"Commit {commit_hash} is still in worktree {self._path}"
            )
        return f"Applied commit {commit_hash[:12]} from worktree to working branch."

    async def discard(self) -> None:
        """Remove the worktree without merging any changes."""
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "remove", "--force", str(self._path),
            cwd=str(self._repo),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        self._active = False
        log.info(f"worktree removed: {self._path}")

    async def status(self) -> str:
        proc = await asyncio.create_subprocess_exec(
            "git", "status", "--short",
            cwd=str(self._path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        changed = stdout.decode(errors="replace").strip()
        n = len([ln for ln in changed.splitlines() if ln.strip()])
        header = f"Worktree: {self._path}  ({n} file(s) changed)"
        return f"{header}\n{changed}" if changed else header
