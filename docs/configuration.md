# Configuration reference

Configuration is TOML, merged in this order (later wins):

1. Built-in defaults (shown below)
2. `~/.config/localcode/config.toml` (user)
3. `<project>/.localcode.toml` (project)
4. `--config FILE` (explicit)

Environment overrides: `LOCALCODE_MODEL`, `LOCALCODE_BASE_URL`,
`LOCALCODE_PROVIDER`.

## Full annotated example

```toml
[provider]
type = "ollama"            # "ollama" (native API) | "openai" (LM Studio, vLLM,
                           #   llama.cpp server, any OpenAI-compatible endpoint)
base_url = "http://localhost:11434"
api_key = ""               # only for servers that require one
model = "qwen2.5-coder:14b"
temperature = 0.2
max_tokens = 4096          # per-response generation cap
context_length = 32768     # context window; drives Ollama num_ctx and compaction
timeout_seconds = 300.0
tool_mode = "auto"         # "auto": native tool API, fall back to JSON parsing
                           # "native": require the server's tool API
                           # "json": always use <tool_call> prompt-based calling
[provider.extra_headers]   # sent with every request
# "X-My-Header" = "value"
[provider.options]         # passed straight to the backend
# top_p = 0.9              # (Ollama options / OpenAI body fields)

# Named profiles, switchable at runtime with /model <name>
[providers.fast]
type = "ollama"
model = "qwen2.5-coder:7b"

[agent]
max_tool_rounds = 40       # hard cap on tool iterations per turn
max_retries = 3            # provider retries (exponential backoff)
retry_base_delay = 1.0
auto_test = true           # run the detected test command after edits
fix_iterations = 3         # repair attempts when those tests fail
verify_edits = true        # re-check files on disk after write/edit
parallel_tools = true      # run read-only tool batches concurrently

[context]
compact_threshold = 0.8    # compact at this fraction of context_length
keep_recent_messages = 8   # never summarized
repo_map_tokens = 2048     # budget for the repo map in the system prompt
max_tool_result_tokens = 4000  # larger tool outputs are middle-truncated

[index]
enabled = true
max_file_size = 1000000    # bytes; larger files are skipped
ignore = ["node_modules/", ".venv/", "dist/", "*.min.js"]  # + .gitignore
use_tree_sitter = true     # needs: uv tool install 'localcode[treesitter]'

[permissions]
approval_mode = "ask"      # "plan" | "ask" | "auto-edit" | "full-auto"
allow_commands = [         # bash prefixes that never prompt
  "git status", "git diff", "git log", "ls", "cat", "pwd", "which", "grep", "find",
]
allow_tools = []           # tool names that never prompt, e.g. ["run_tests"]
deny_commands = [          # always blocked, even in full-auto
  "rm -rf /", "sudo rm", "mkfs", "shutdown", "reboot",
]

[cache]
enabled = true
directory = ""             # default: ~/.cache/localcode
prompt_cache = true        # keep prompt prefixes stable for server KV caching
index_ttl_seconds = 0      # 0 = invalidate by mtime only

[logging]
level = "INFO"             # DEBUG | INFO | WARNING | ERROR
console = false            # console logging corrupts the TUI; file-only default
directory = ""             # default: ~/.local/share/localcode/logs

[api]                      # for --serve
host = "127.0.0.1"
port = 8765

plugins = []               # extra plugin module paths, e.g. ["mytools.localcode"]
```

## Backend cheat sheet

| Backend | type | base_url | Notes |
|---|---|---|---|
| Ollama | `ollama` | `http://localhost:11434` | native API; `num_ctx` set from `context_length` |
| LM Studio | `openai` | `http://localhost:1234` | enable the local server in LM Studio |
| llama.cpp | `openai` | `http://localhost:8080` | `llama-server -m model.gguf -c 32768` |
| vLLM | `openai` | `http://localhost:8000` | add `--enable-auto-tool-choice` for native tools |
| Other OpenAI-compatible | `openai` | your endpoint | set `api_key` if required |

## Model notes

- **Qwen2.5-Coder** (7B/14B/32B) — best tool-calling reliability; `auto` mode
  uses native tools on Ollama and vLLM.
- **DeepSeek-Coder-V2** — native tools on recent Ollama.
- **Codestral** — native tools.
- **Llama 3.x** — native tools.
- **Gemma** — no native tool API; localcode automatically falls back to
  `<tool_call>` JSON parsing (or set `tool_mode = "json"` explicitly).

## File locations

| What | Where |
|---|---|
| User config | `~/.config/localcode/config.toml` |
| Plugins | `~/.config/localcode/plugins/*.py` |
| Sessions & checkpoints | `~/.local/share/localcode/` |
| Logs | `~/.local/share/localcode/logs/localcode.log` |
| Index cache | `~/.cache/localcode/index/` |

All respect `XDG_CONFIG_HOME` / `XDG_DATA_HOME` / `XDG_CACHE_HOME`.
