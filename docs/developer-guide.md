# Developer guide

## Setup

```bash
git clone <repo-url> && cd localcode
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,treesitter,api]"
```

## Everyday commands

```bash
.venv/bin/pytest                       # test suite (fast, no network, no model)
.venv/bin/pytest --cov=localcode       # with coverage
.venv/bin/mypy localcode               # strict type checking (must be clean)
.venv/bin/ruff check localcode tests   # lint (must be clean)
.venv/bin/ruff format localcode tests  # formatting
python benchmarks/run_benchmarks.py    # benchmark suite
```

## Code map

```
localcode/
  config/       schema.py (pydantic models)  loader.py (TOML layering)
  providers/    base.py (Provider ABC, events, factory)
                openai_compat.py  ollama.py  toolparse.py (JSON fallback)
  tools/        base.py (Tool ABC, registry, ToolContext)
                fs.py search.py shell.py gittools.py refactor.py repo.py
  index/        detect.py imports.py graph.py repomap.py
  context/      manager.py (token budget, compaction)
  agent/        loop.py (the agentic loop)  prompts.py  checkpoints.py
  session/      store.py (JSONL persistence)
  cli/          app.py render.py commands.py approval.py
  api/          server.py (optional FastAPI)
  plugins/      loader.py
tests/          mirrors the package; FakeProvider in conftest.py
benchmarks/     mock_server.py (scripted OpenAI-compat server)  run_benchmarks.py
```

## Testing philosophy

- **No model required.** `FakeProvider` (tests/conftest.py) replays scripted
  `(text, tool_calls)` turns through the real agent loop. HTTP providers are
  tested against `httpx.MockTransport`. End-to-end tests use
  `benchmarks/mock_server.py`, a real HTTP server that behaves like a
  scripted agent.
- Every tool has direct tests against a real temp workspace, including
  failure paths (missing files, ambiguous edits, timeouts, path escapes).
- Agent behaviors (approval gating, plan mode, deny list, auto-test repair
  loop, compaction triggering) are tested as event-stream assertions.

## Adding a tool

```python
# localcode/tools/mytool.py
from localcode.tools.base import Tool, ToolResult

class CountLinesTool(Tool):
    name = "count_lines"
    description = "Count lines in a file."
    permission = "read"                      # read | write | execute
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def run(self, path: str, **_) -> ToolResult:
        target = self.ctx.resolve(path)      # workspace-rooted, escape-safe
        n = len(target.read_text().splitlines())
        return ToolResult(ok=True, output=f"{n} lines", data={"lines": n})

TOOLS = [CountLinesTool]
```

Rules:
- `write`-permission tools must call `self.ctx.before_mutation(path)` before
  changing any file (this powers checkpoints and `/undo`).
- Return `ToolResult.error("...")` for expected failures; the model reads the
  message and adapts. Raise only for programming errors.
- Keep `output` concise; put machine-readable detail in `data`.

Register built-ins in `tools/base.py:build_default_registry`, or ship the
tool as a plugin (see [plugins.md](plugins.md)).

## Adding a provider

Subclass `Provider` (`providers/base.py`) and implement:

- `_chat_native(messages, tools) -> AsyncIterator[StreamEvent]` — raise
  `ProviderError(retryable=True)` for transient transport errors.
- `list_models()`, `close()`.

Add the type to `create_provider()` and to `ProviderConfig.type`. The base
class supplies tool-mode selection, the JSON fallback, and `complete()` —
you only speak the wire protocol.

## Design invariants (please keep)

1. The system prompt must stay byte-identical between turns (prompt cache).
2. Conversation history is append-only except compaction.
3. Never split an assistant tool_call from its tool result in compaction.
4. Tool crashes must never escape the agent loop — they become error results.
5. The deny list is checked before every approval path, including full-auto.
6. All user-facing behavior belongs in `Config`, not in constants.

## Release checklist

pytest green · mypy clean · ruff clean · benchmarks not regressed ·
docs updated · version bumped in `pyproject.toml` and `localcode/__init__.py`.
