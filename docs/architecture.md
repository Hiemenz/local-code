# Architecture

localcode is organized as ten independent modules with one-way dependencies.
Every module can be used as a library without the CLI.

```
┌─────────────────────────────────────────────────────────────┐
│  cli/            REPL, rendering, slash commands, approvals │
│  api/            optional HTTP server (NDJSON streaming)    │
├─────────────────────────────────────────────────────────────┤
│  agent/          the agentic loop, prompts, checkpoints     │
├──────────────┬──────────────┬───────────────┬───────────────┤
│  providers/  │  tools/      │  context/     │  session/     │
│  LLM backends│  tool system │  compaction   │  persistence  │
├──────────────┴──────────────┴───────────────┴───────────────┤
│  index/          repository intelligence                    │
├─────────────────────────────────────────────────────────────┤
│  config/  logging.py  errors.py  plugins/                   │
└─────────────────────────────────────────────────────────────┘
```

## Module responsibilities

### `config/`
Pydantic schema (`schema.py`) and layered TOML loading (`loader.py`).
Precedence: built-in defaults → `~/.config/localcode/config.toml` →
`<project>/.localcode.toml` → `--config FILE`. Every behavior is a config
field; nothing is hard-coded.

### `providers/`
One abstraction (`Provider`) over all backends. `Provider.chat()` streams
`StreamEvent`s (`text`, `tool_call`, `usage`).

- `openai_compat.py` — SSE `/v1/chat/completions`; covers LM Studio, vLLM,
  llama.cpp server, and anything OpenAI-shaped.
- `ollama.py` — native NDJSON `/api/chat`; gives `num_ctx` control, model
  listing, and `keep_alive`.
- `toolparse.py` — the **JSON tool-call fallback**. Models without a native
  tool API get tool schemas injected into the system prompt and emit
  `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks (the format
  Qwen was trained on). `StreamToolExtractor` strips these from the token
  stream incrementally, even when a block spans chunks, so users never see
  raw JSON. With `tool_mode = "auto"` (default) the provider tries the native
  API first and permanently falls back on the first tools-related failure.

Retry policy: connection failures and 5xx are `retryable`; the agent retries
with exponential backoff, but never after partial output has been shown.

### `tools/`
Each tool declares `name`, `description`, a JSON-schema `parameters`, and a
`permission` (`read` / `write` / `execute`). Results are structured
(`ToolResult.ok/output/data`). `ToolContext` provides workspace-rooted path
resolution (escapes are rejected), checkpoint hooks, and the repo index.
`build_default_registry()` registers the 21 built-ins plus plugins.

### `index/`
- `detect.py` — languages by extension; build systems, package managers, and
  test frameworks by marker files (incl. `package.json` script parsing).
  Produces the project's canonical `test_command`/`build_command`.
- `imports.py` — imports + top-level symbols per file. Tree-sitter when the
  `treesitter` extra is installed; per-language regex fallback otherwise.
- `graph.py` — resolves import specs to repo files (Python dotted paths, JS
  relative paths with index resolution, Rust mods, C includes) and ranks
  files with PageRank: files that many others depend on rank highest.
- `repomap.py` — `RepoIndex` walks the tree (honoring `.gitignore` +
  configured ignores), re-parses only files whose mtime/size changed
  (persistent JSON cache keyed by repo path), and renders a token-budgeted
  repo map for the system prompt. `relevant_files(query)` combines graph
  rank with query-term matches on paths and symbols.

### `context/`
Token estimation (chars/4 — tokenizer-free and adequate for budgeting) and
compaction. When the conversation exceeds `compact_threshold ×
context_length`, everything before the recent window is summarized by the
model itself into a single message. The split point always lands on a user
message so tool_call/tool_result pairs are never separated. Message order is
otherwise never changed, keeping the prompt prefix stable so server-side
KV/prompt caches keep hitting.

### `agent/`
`loop.py` is the heart: stream a completion → execute tool calls → append
results → repeat, until the model answers without tools. Details:

- **Approval gates** — deny-list checks always run (even in `full-auto`);
  read-only tools never prompt; `ask`/`auto-edit`/`full-auto` control the
  rest; "always" answers add session allowlist entries.
- **Parallel tools** — consecutive all-read-only call batches run under
  `asyncio.gather`; results are re-ordered to match the transcript.
- **Edit verification** — after `write_file`/`edit_file`, the file is
  re-checked on disk; a failed verification is reported to the model.
- **Auto-test-and-fix** — when files changed and the project has a detected
  test command, tests run after the model finishes; failures are pushed back
  as a message for up to `fix_iterations` repair rounds.
- **Checkpoints** (`checkpoints.py`) — every file is snapshotted before its
  first mutation in a turn; `/undo` restores the whole turn atomically.
- **Interrupts** — SIGINT cancels the turn task; partial state is kept and
  the session file already contains everything up to the interruption.

### `session/`
Append-only JSONL transcripts per project (`~/.local/share/localcode/
sessions/<project-hash>/`). `to_messages()` reconstructs provider history
for `--resume`/`--continue`, honoring compaction events.

### `cli/`
`prompt_toolkit` input (history, multiline) + `rich` output. Streaming text
renders as live markdown, throttled to ~12 fps. Slash commands mirror
Claude Code (`/help /clear /compact /cost /model /plan /mode /sessions
/resume /undo /history /diff /map /init /config`). `!cmd` runs a shell
command directly and shares the output with the model.

### `api/`
Optional FastAPI server (`--serve`): `POST /v1/message` streams agent events
as NDJSON; useful for editor integrations.

### `plugins/`
Three discovery paths: the `localcode.tools` entry-point group, module paths
in config, and `~/.config/localcode/plugins/*.py`. A plugin is a
`register(registry, ctx)` function. Broken plugins log a warning and are
skipped.

## Performance decisions

- **Prompt-cache friendliness**: the system prompt (including repo map) is
  built once and kept byte-identical across turns; history is append-only
  except for compaction. Local servers with prefix caching (llama.cpp,
  vLLM, Ollama) therefore skip re-processing the prompt prefix each turn.
- **Incremental indexing**: mtime+size cache means warm scans re-parse only
  changed files; the index refreshes after any turn that modified files.
- **Tool output truncation**: middle-out truncation bounded by
  `max_tool_result_tokens` keeps single results from flooding the context.
- Measured overhead (Raspberry Pi 5): full 3-round agentic turn ≈ 24 ms of
  harness time; cold index of this repo ≈ 20 ms. See `benchmarks/`.
