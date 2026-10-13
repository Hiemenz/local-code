# Installation

## Requirements

- Python 3.12 or newer
- git
- A local LLM server (Ollama recommended)
- Optional: ripgrep (`rg`) for faster search; a pure-Python fallback is built in

## 1. Install a backend and model

### Ollama (recommended)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:14b       # best all-round; needs ~12 GB RAM
# smaller / larger alternatives:
ollama pull qwen2.5-coder:7b        # ~6 GB
ollama pull deepseek-coder-v2:16b
ollama pull codestral
ollama pull llama3.1:8b
ollama pull gemma2:9b               # no native tools; JSON fallback used automatically
```

### LM Studio
Start the local server (default `http://localhost:1234`). Set
`provider.type = "openai"` and `base_url` accordingly.

### llama.cpp
```bash
llama-server -m model.gguf --port 8080 -c 32768
```
Use `provider.type = "openai"`, `base_url = "http://localhost:8080"`.

### vLLM
```bash
vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
```
Use `provider.type = "openai"`, `base_url = "http://localhost:8000"`.

## 2. Install localcode

From PyPI (when published) or from source:

```bash
git clone <repo-url> localcode && cd localcode
uv sync
# optional extras:
uv sync --extra treesitter   # precise symbol extraction
uv sync --extra api          # HTTP API server
```

`uv tool install .` also works and puts `lc` / `localcode` on your PATH.

## 3. First run

```bash
lc ~/path/to/your/repo
```

With Ollama running and `qwen2.5-coder:14b` pulled, no configuration is
needed. Otherwise create `~/.config/localcode/config.toml`:

```toml
[provider]
type = "ollama"                      # or "openai" for everything else
base_url = "http://localhost:11434"
model = "qwen2.5-coder:14b"
```

See [configuration.md](configuration.md) for every option.

## Docker

```bash
docker compose up -d          # starts Ollama + localcode
docker compose exec ollama ollama pull qwen2.5-coder:14b
docker compose run --rm localcode lc /workspace
```

Mount the repository you want to work on at `/workspace` (see
`docker-compose.yml`). Or standalone:

```bash
docker build -t localcode .
docker run -it --rm \
  -v "$PWD":/workspace \
  -e LOCALCODE_BASE_URL=http://host.docker.internal:11434 \
  localcode /workspace
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection to ... failed. Is 'ollama serve' running?` | Start the backend; check `base_url` |
| Model gives text but never calls tools | Set `provider.tool_mode = "json"` |
| Responses truncated | Raise `provider.max_tokens` |
| Model forgets earlier context | Raise `provider.context_length` (and Ollama `num_ctx` is set from it automatically) |
| Slow first token each turn | Use a backend with prefix caching enabled; keep `cache.prompt_cache = true` |
