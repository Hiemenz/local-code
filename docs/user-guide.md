# User guide

If you know Claude Code, you already know localcode. This guide covers the
workflow and where the two differ.

## Starting

```bash
lc                      # current directory
lc ~/src/myproject      # a specific repository
lc --continue           # pick up the most recent session
lc --resume             # resume latest (or /sessions + /resume <id> inside)
lc --mode plan          # start in read-only plan mode
lc -p "explain the build system"   # one-shot, non-interactive, prints and exits
```

On startup localcode indexes the repository (incremental — near-instant
after the first run), detects the project type, and injects a ranked repo
map into the model's context.

## The interaction loop

Type what you want in plain language:

```
› the date parser breaks on ISO week numbers, fix it and add a test
```

The agent will search/read the relevant files, propose edits, run them
past your approval gate, apply them, run the test suite, and fix any
failures it introduced — then summarize what changed.

- **Enter** submits. **Alt+Enter** inserts a newline (pasted newlines also work).
- **Ctrl+C** interrupts a running turn (state is kept; just keep typing).
- **Ctrl+D** or `/quit` exits.
- **`!command`** runs a shell command directly; its output is shared with
  the model as context.
- **↑/↓** browse input history (persisted across sessions).

## Approval modes

| Mode | Reads | Edits | Commands |
|---|---|---|---|
| `plan` | ✓ | blocked | blocked |
| `ask` (default) | ✓ | prompt | prompt |
| `auto-edit` | ✓ | ✓ | prompt |
| `full-auto` | ✓ | ✓ | ✓ (deny list still enforced) |

Switch anytime with `/mode auto-edit` or toggle plan mode with `/plan`.
Approval prompts show a unified diff for edits and the exact command for
shell calls; answering **a** (always) allowlists that tool or program for
the session. Permanent allowlists live in config
(`permissions.allow_commands`, `permissions.allow_tools`).

Safe commands (`git status`, `ls`, `grep`, …) never prompt. Destructive
patterns (`sudo rm`, `mkfs`, …) are always blocked, even in `full-auto`.

## Plan mode

`/plan` restricts the agent to read-only tools and instructs it to produce
an implementation plan instead of editing. Review the plan, then `/plan`
again (back to `ask`) and say "go ahead".

## Checkpoints and undo

Every file is snapshotted before its first modification in each turn.

- `/history` — list turns that changed files
- `/undo` — revert the last turn's file changes (repeatable)

Files *created* by a turn are deleted on undo; modified files are restored
byte-for-byte.

## Sessions

Everything is persisted as it happens; a crash or Ctrl+C loses nothing.

- `/sessions` — list sessions for this project
- `/resume <id>` — switch to an old session (full history restored)
- `lc --continue` — resume the latest session at launch

## Context management

- `/cost` — token usage and current context size
- `/compact` — manually summarize older history
- Automatic compaction triggers at 80% of the context window (configurable);
  the model writes a dense summary and the conversation continues seamlessly.

## Long-running autonomous tasks

For big refactors, combine `full-auto` with auto-testing:

```bash
lc --mode full-auto
› migrate all datetime.utcnow() calls to timezone-aware now(); run tests when done
```

The agent loops through edits and test runs (up to `agent.fix_iterations`
repair attempts) before returning control. `agent.max_tool_rounds` caps
runaway loops. Interrupt at any time with Ctrl+C; `/undo` reverts a bad turn.

## Repository intelligence

- `/map` — the ranked repository map the model sees
- `/init` — force a re-index (also refreshes the system prompt)
- The model can call `find_relevant_files` and `repo_map` itself.

## Git workflow

Ask in plain language ("commit this as 'fix parser'", "create a branch
first", "show me the diff"), or use `/diff` to review working-tree changes
yourself. The agent never commits unless you explicitly ask.

## Switching models

- `/model` — show the current model and list what the backend has available
- `/model qwen2.5-coder:32b` — switch model in place
- `/model fast` — switch to a named provider profile from config:

```toml
[providers.fast]
type = "ollama"
model = "qwen2.5-coder:7b"

[providers.big]
type = "openai"
base_url = "http://gpu-box:8000"
model = "Qwen/Qwen2.5-Coder-32B-Instruct"
```

## Project instructions (LOCALCODE.md)

If the repository root contains `LOCALCODE.md`, `CLAUDE.md`, or `AGENTS.md`
(checked in that order), its contents are injected into the system prompt as
repository-specific instructions — exactly like Claude Code's `CLAUDE.md`.
Existing repositories with a `CLAUDE.md` work unchanged. Use it for build
quirks, style rules, and "always run X before committing" guidance.

## All slash commands

`/help` `/clear` `/compact` `/config` `/cost` `/diff` `/exit` `/history`
`/init` `/map` `/mode` `/model` `/plan` `/quit` `/resume` `/sessions`
`/tools` `/undo`
