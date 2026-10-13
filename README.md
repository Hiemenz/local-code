# localcode

**A Claude Code-compatible coding assistant that runs entirely on local LLMs.**

localcode reproduces the Claude Code workflow — interactive terminal UI, streaming
responses, agentic planning, automatic tool use, file editing, git integration,
checkpoints, session resume, and approval gates — with all inference on your own
hardware via Ollama, LM Studio, llama.cpp, vLLM, or any OpenAI-compatible server.

```
$ lc ~/my-project
localcode v0.1.0 · qwen2.5-coder:14b via ollama @ http://localhost:11434
/home/you/my-project · 214 files indexed · mode: ask · session: 20260709-101502-a3f9c1

› fix the failing test in tests/test_parser.py
  ⏺ run_tests({"args": "tests/test_parser.py"})
  ⎿ $ python -m pytest tests/test_parser.py
  ⏺ read_file({"path": "src/parser.py"})
  ⎿      1  import re ...
  ⏺ edit_file({"path": "src/parser.py", ...})
  ⎿ Replaced 1 occurrence(s) in src/parser.py
· Tests passed.
The bug was an off-by-one in parse_range (src/parser.py:88) — fixed and verified.
```

## Quick start

```bash
# 1. Install Ollama and pull a coding model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:14b

# 2. Install localcode
uv tool install localcode

# 3. Point it at a repository
lc ~/path/to/repo
```

No configuration is required for the Ollama + qwen2.5-coder default. To change
model or backend, edit `~/.config/localcode/config.toml` — see
[docs/configuration.md](docs/configuration.md).

## Features

- **Interactive terminal UI** with streamed, markdown-rendered responses
- **Agentic loop**: plans, reads code, edits, runs tests, fixes failures before returning
- **22 built-in tools**: file read/write/edit, atomic multi-edit, unified-diff
  patching, grep/ripgrep (with context lines), glob, shell, Python/Node execution,
  tests, builds, git status/diff/commit/branch/log, cross-file symbol rename,
  repo map, relevance search
- **Project instructions**: `LOCALCODE.md` / `CLAUDE.md` / `AGENTS.md` at the repo
  root is injected into the system prompt, like Claude Code's CLAUDE.md
- **Repository intelligence**: incremental index, import graph + PageRank ranking,
  build-system/package-manager/test-framework detection, tree-sitter parsing (optional)
- **Approval gates**: `plan` / `ask` / `auto-edit` / `full-auto` modes, per-command
  allowlists, hard deny list, diff previews before every edit
- **Checkpoints**: every turn's file changes are snapshotted — `/undo` reverts them
- **Sessions**: persisted as JSONL; `--continue`, `--resume`, `/sessions`
- **Context management**: automatic compaction via model summarization, `/compact`, `/cost`
- **Any local backend**: Ollama (native), LM Studio, llama.cpp server, vLLM, or any
  OpenAI-compatible endpoint; models without native tool-calling work via a
  `<tool_call>` JSON fallback
- **Plugins**: add tools with a single `register()` function
- **HTTP API** (optional): drive the agent over NDJSON streaming

## Supported models

Any chat model works; these are tested defaults:

| Model | Suggested size | Tool calling |
|---|---|---|
| Qwen2.5-Coder | 7B / 14B / 32B | native |
| DeepSeek-Coder-V2 | 16B | native |
| Codestral | 22B | native |
| Llama 3.x | 8B / 70B | native |
| Gemma 2/3 | 9B / 27B | JSON fallback (automatic) |

## Documentation

- [Installation guide](docs/installation.md)
- [User guide](docs/user-guide.md)
- [Configuration reference](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [Developer guide](docs/developer-guide.md)
- [Plugin authoring](docs/plugins.md)
- [Benchmarks](docs/benchmarks.md)

## Development

```bash
git clone <repo> && cd localcode
uv sync
uv run pytest                 # test suite
uv run mypy localcode         # type checking
uv run ruff check localcode   # lint
```

## License

MIT
