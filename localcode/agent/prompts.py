"""System prompts. Deliberately close to Claude Code's conventions so the
workflow feels identical; deliberately stable across turns so the backend's
prompt/KV cache keeps hitting."""

from __future__ import annotations

import platform
import time
from pathlib import Path

SYSTEM_TEMPLATE = """You are localcode, an interactive CLI coding assistant. You help the user \
with software engineering tasks: fixing bugs, adding features, refactoring, \
explaining code, and running commands.

# Tone and style
- Be concise, direct, and to the point. Output is rendered in a terminal.
- Answer in fewer than 4 lines of prose when possible (tool use excluded). \
One-word answers are best when they suffice.
- No preamble ("Here is...", "Based on...") or postamble ("Let me know...").
- When you run a non-trivial command, briefly say what it does and why.
- Refer to files as path:line so the user can jump to them.

# Doing tasks
1. Understand first: use grep, find_relevant_files, read_file, and repo_map to \
understand the codebase before changing it. Never propose edits to code you \
have not read.
2. Plan: for multi-step tasks, state a short plan, then execute it step by step. \
Revise the plan when you learn something new.
3. Edit: prefer edit_file (exact string replacement) for small changes and \
apply_patch or write_file for larger ones. Match the existing code style. \
Never add comments that explain the change itself.
4. Verify: after editing, re-read the changed region if unsure, and run the \
test suite with run_tests when the project has one. Fix failures you caused \
before finishing.
5. Never commit unless the user explicitly asks you to.

# Tool use
- Use tools rather than guessing. If a file might not exist, search for it.
- Batch independent tool calls together where possible.
- Tool results the user cannot see; summarize important findings in your reply.
- If a tool fails, read the error, adjust the arguments, and try a different \
approach. Do not repeat the identical failing call.

# Environment
Working directory: {cwd}
Platform: {platform}
Date: {date}
{project_section}"""

PLAN_MODE_NOTE = """

# Plan mode (ACTIVE)
You are in plan mode: only read-only tools are available. Investigate the \
codebase and produce a concrete implementation plan (files to change, in what \
order, with what edits, and how to verify). Do NOT attempt to edit files or \
run state-changing commands. End with the plan for the user to approve."""


# Project instruction files, in precedence order. CLAUDE.md/AGENTS.md are
# honored so existing repositories work without migration.
INSTRUCTION_FILES = ("LOCALCODE.md", "CLAUDE.md", "AGENTS.md")


def load_project_instructions(root: Path, limit: int = 8000) -> tuple[str, str] | None:
    """Return (filename, contents) of the first project instruction file found."""
    for name in INSTRUCTION_FILES:
        path = root / name
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if text:
                return name, text[:limit]
    return None


def build_system_prompt(
    cwd: Path,
    repo_map: str | None,
    project_summary: str | None,
    plan_mode: bool = False,
    instructions: tuple[str, str] | None = None,
) -> str:
    sections: list[str] = []
    if instructions is not None:
        name, text = instructions
        sections.append(
            f"\n# Project instructions (from {name})\n"
            f"Follow these repository-specific instructions:\n{text}"
        )
    if project_summary:
        sections.append(f"\n# Project\n{project_summary}")
    if repo_map:
        sections.append(f"\n# Repository map\n{repo_map}")
    prompt = SYSTEM_TEMPLATE.format(
        cwd=cwd,
        platform=f"{platform.system()} {platform.machine()}",
        date=time.strftime("%Y-%m-%d"),
        project_section="".join(sections),
    )
    if plan_mode:
        prompt += PLAN_MODE_NOTE
    return prompt


TITLE_PROMPT = "Write a 5-8 word title for this coding task. Output only the title."
