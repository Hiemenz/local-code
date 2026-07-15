# Benchmarks

The benchmark suite (`benchmarks/run_benchmarks.py`) measures **localcode's
own overhead**, isolated from model inference, by replaying scripted
responses from a mock OpenAI-compatible server
(`benchmarks/mock_server.py`). This answers "how much latency does the tool
add on top of the model?" — for a local-LLM assistant, the model dominates
everything else, and the harness must stay negligible.

```bash
python benchmarks/run_benchmarks.py              # print results
python benchmarks/run_benchmarks.py --json out.json
```

## What is measured

| Group | Metric |
|---|---|
| Indexing | cold scan, warm (incremental) scan, repo-map render, relevance query |
| Tools | median latency of read_file, grep, glob, bash, edit_file |
| Agent | full 3-round agentic turn (2 tool calls + streamed reply) through the real provider/agent/tool stack; 200-chunk stream throughput |
| Real backend (optional) | time-to-first-token, total time, tokens/sec |

## Reference results

Raspberry Pi 5 (ARM Cortex-A76, Python 3.13) — i.e. close to a worst-case
host; expect several times faster on a desktop:

```
[1] Repository indexing (this repo, 57 files)
  index_cold_scan                 20.4 ms
  index_warm_scan(incremental)     8.6 ms
  render_repo_map                  0.1 ms
  relevance_ranking                0.1 ms

[2] Tool execution latency (median of 5)
  read_file                        0.4 ms
  grep(regex, whole tree)          3.5 ms
  glob                             2.6 ms
  bash(echo)                       0.6 ms
  edit_file                        0.1 ms

[3] Agent harness overhead
  end_to_end_turn(2 tool calls + reply)   23.6 ms
  stream_200_chunks_through_stack          5.4 ms
```

Interpretation: a full agentic turn costs ~24 ms of harness time. A 14B
model generating 100 tokens at 25 tok/s takes ~4000 ms — localcode adds
well under 1% of end-to-end latency.

## Benchmarking a real backend

```bash
LOCALCODE_BENCH_BASE_URL=http://localhost:11434 \
LOCALCODE_BENCH_MODEL=qwen2.5-coder:14b \
LOCALCODE_BENCH_PROVIDER=ollama \
python benchmarks/run_benchmarks.py
```

Adds time-to-first-token, total response time, and tokens/sec for your
hardware/model combination — useful when choosing a model size.

## Large-repository behavior

Indexing is O(changed files) after the first scan thanks to the mtime/size
cache; the cold scan is bounded by `index.max_file_size` and the ignore
rules. On repos in the tens of thousands of files, tune:

```toml
[index]
ignore = ["vendor/", "third_party/", "*.generated.*"]

[context]
repo_map_tokens = 1024   # keep the map lean so prompts stay cacheable
```
