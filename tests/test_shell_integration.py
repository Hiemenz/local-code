"""Tests for localcode.shell.integration."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from localcode.shell.integration import (  # noqa: E402
    _project_sha1,
    clear_active,
    write_active,
    zsh_script,
)

# ---------------------------------------------------------------------------
# _project_sha1
# ---------------------------------------------------------------------------

def test_sha1_matches_hashlib() -> None:
    root = "/home/user/myproject"
    expected = hashlib.sha1(root.encode()).hexdigest()[:16]
    assert _project_sha1(root) == expected


def test_sha1_length() -> None:
    assert len(_project_sha1("/any/path")) == 16


def test_sha1_different_paths_differ() -> None:
    assert _project_sha1("/a/b") != _project_sha1("/a/c")


# ---------------------------------------------------------------------------
# write_active / clear_active
# ---------------------------------------------------------------------------

def test_write_active_creates_file(tmp_path: Path) -> None:
    write_active(tmp_path, "/proj/foo", "sess-123")
    sha1 = _project_sha1("/proj/foo")
    state_file = tmp_path / "active" / sha1
    assert state_file.exists()
    assert state_file.read_text() == "sess-123"


def test_write_active_creates_parent_dirs(tmp_path: Path) -> None:
    base = tmp_path / "deep" / "nested"
    write_active(base, "/proj", "s1")
    assert (base / "active" / _project_sha1("/proj")).exists()


def test_write_active_overwrites(tmp_path: Path) -> None:
    write_active(tmp_path, "/proj", "old-id")
    write_active(tmp_path, "/proj", "new-id")
    sha1 = _project_sha1("/proj")
    assert (tmp_path / "active" / sha1).read_text() == "new-id"


def test_clear_active_removes_file(tmp_path: Path) -> None:
    write_active(tmp_path, "/proj", "sess-abc")
    clear_active(tmp_path, "/proj")
    sha1 = _project_sha1("/proj")
    assert not (tmp_path / "active" / sha1).exists()


def test_clear_active_missing_file_is_silent(tmp_path: Path) -> None:
    # Should not raise when file doesn't exist
    clear_active(tmp_path, "/proj/nonexistent")


def test_separate_projects_have_separate_files(tmp_path: Path) -> None:
    write_active(tmp_path, "/proj/a", "sess-a")
    write_active(tmp_path, "/proj/b", "sess-b")
    sha_a = _project_sha1("/proj/a")
    sha_b = _project_sha1("/proj/b")
    assert sha_a != sha_b
    assert (tmp_path / "active" / sha_a).read_text() == "sess-a"
    assert (tmp_path / "active" / sha_b).read_text() == "sess-b"

    clear_active(tmp_path, "/proj/a")
    assert not (tmp_path / "active" / sha_a).exists()
    assert (tmp_path / "active" / sha_b).exists()


# ---------------------------------------------------------------------------
# zsh_script
# ---------------------------------------------------------------------------

def test_zsh_script_is_non_empty() -> None:
    script = zsh_script()
    assert len(script) > 100


def test_zsh_script_defines_lc_function() -> None:
    assert "lc()" in zsh_script()


def test_zsh_script_has_precmd_hook() -> None:
    assert "precmd_functions" in zsh_script()
    assert "_lc_precmd" in zsh_script()


def test_zsh_script_references_sha1_command() -> None:
    assert "shasum" in zsh_script()


def test_zsh_script_references_lc_indicator() -> None:
    assert "_lc_indicator" in zsh_script()


def test_zsh_script_injects_shell_branch() -> None:
    assert "LOCALCODE_SHELL_BRANCH" in zsh_script()


def test_zsh_script_has_continue_flag() -> None:
    # The wrapper should inject --continue for auto-resume
    assert "--continue" in zsh_script()


# ---------------------------------------------------------------------------
# CLI --shell-integration flag
# ---------------------------------------------------------------------------

def test_shell_integration_flag_prints_and_exits(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["localcode", "--shell-integration"])
    from localcode.cli.app import main
    main()  # should return without raising SystemExit
    out = capsys.readouterr().out
    assert "lc()" in out
    assert "_lc_precmd" in out


def test_shell_integration_flag_no_app_construction(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --shell-integration must exit before building App (no config needed)
    monkeypatch.setattr(sys, "argv", ["localcode", "--shell-integration"])
    from localcode.cli.app import main
    main()
    out = capsys.readouterr().out
    assert len(out) > 0
