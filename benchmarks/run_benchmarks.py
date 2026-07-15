"""localcode benchmark suite.

Measures the tool's own overhead (indexing, tools, agent harness) using a
scripted mock backend, so results isolate localcode from model speed.
Optionally benchmarks a real backend when LOCALCODE_BENCH_BASE_URL is set.

Run: python benchmarks/run_benchmarks.py [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mock_server import start as start_mock  # noqa: E402

from localcode.agent.loop import Agent  # noqa: E402
from localcode.config.schema import Config, IndexConfig, ProviderConfig  # noqa: E402
from localcode.index.repomap import RepoIndex  # noqa: E402
from localcode.providers.base import create_provider  # noqa: E402
from localcode.tools.base import ToolContext, build_default_registry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS: dict[str, float] = {}


def timed(name: str, seconds: float) -> None:
    RESULTS[name] = round(seconds, 4)
    print(f"  {name:<44} {seconds * 1000:>10.1f} ms")


def bench_indexing() -> None:
    print("\n[1] Repository indexing (this repo)")
    with tempfile.TemporaryDirectory() as cache:
        index = RepoIndex(REPO_ROOT, IndexConfig(), Path(cache))
        t0 = time.perf_counter()
        index.scan()
        timed("index_cold_scan", time.perf_counter() - t0)
        t0 = time.perf_counter()
        index.scan()
        timed("index_warm_scan(incremental)", time.perf_counter() - t0)
        t0 = time.perf_counter()
        index.render_map(8000)
        timed("render_repo_map", time.perf_counter() - t0)
        t0 = time.perf_counter()
        index.relevant_files("provider streaming tool call")
        timed("relevance_ranking", time.perf_counter() - t0)
        print(f"      ({len(index.files)} files indexed)")


async def bench_tools() -> None:
    print("\n[2] Tool execution latency (median of 5)")
    with tempfile.TemporaryDirectory() as workdir:
        root = Path(workdir)
        shutil.copytree(REPO_ROOT / "localcode", root / "localcode")
        ctx = ToolContext(root=root, config=Config())
        registry = build_default_registry(ctx)

        async def median(name: str, **kwargs: object) -> float:
            tool = registry.get(name)
            assert tool is not None
            samples = []
            for _ in range(5):
                t0 = time.perf_counter()
                result = await tool.run(**kwargs)
                samples.append(time.perf_counter() - t0)
                assert result.ok, result.output
            return statistics.median(samples)

        timed("read_file", await median("read_file", path="localcode/agent/loop.py"))
        timed("grep(regex, whole tree)", await median("grep", pattern="class \\w+Provider"))
        timed("glob", await median("glob", pattern="**/*.py"))
        timed("bash(echo)", await median("bash", command="echo hi"))
        timed(
            "edit_file",
            await median(
                "edit_file", path="localcode/errors.py",
                old_string="Base class for all localcode errors.",
                new_string="Base class for all localcode errors.",  # noop content, real IO
                replace_all=True,
            ),
        )


async def bench_agent_harness() -> None:
    print("\n[3] Agent harness overhead (scripted 3-round task, mock backend)")
    server, port = start_mock()
    try:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            (root / "main.py").write_text("def greet():\n    return 'hi'\n")
            config = Config()
            config.agent.auto_test = False
            config.provider = ProviderConfig(
                type="openai", base_url=f"http://127.0.0.1:{port}", model="mock-coder"
            )
            ctx = ToolContext(root=root, config=config)
            provider = create_provider(config.provider)
            agent = Agent(
                provider=provider,
                registry=build_default_registry(ctx),
                tool_ctx=ctx,
                config=config,
            )
            agent.mode = "full-auto"
            agent.set_system_prompt("bench")
            t0 = time.perf_counter()
            async for _event in agent.run_turn("change greet to return hello world"):
                pass
            timed("end_to_end_turn(2 tool calls + reply)", time.perf_counter() - t0)
            assert "hello world" in (root / "main.py").read_text()

            t0 = time.perf_counter()
            async for _event in agent.run_turn("BENCH:200"):
                pass
            timed("stream_200_chunks_through_stack", time.perf_counter() - t0)
            await provider.close()
    finally:
        server.shutdown()


async def bench_real_backend() -> None:
    base_url = os.environ.get("LOCALCODE_BENCH_BASE_URL")
    if not base_url:
        print("\n[4] Real backend: skipped (set LOCALCODE_BENCH_BASE_URL to enable)")
        return
    model = os.environ.get("LOCALCODE_BENCH_MODEL", "qwen2.5-coder:14b")
    ptype = os.environ.get("LOCALCODE_BENCH_PROVIDER", "ollama")
    print(f"\n[4] Real backend: {model} @ {base_url}")
    cfg = ProviderConfig(type=ptype, base_url=base_url, model=model)  # type: ignore[arg-type]
    provider = create_provider(cfg)
    from localcode.providers.base import Message

    t0 = time.perf_counter()
    first_token: float | None = None
    tokens = 0
    async for event in provider.chat(
        [Message(role="user", content="Count from 1 to 50, comma separated.")]
    ):
        if event.kind == "text" and first_token is None:
            first_token = time.perf_counter() - t0
        if event.kind == "usage" and event.usage is not None:
            tokens = event.usage.completion_tokens
    total = time.perf_counter() - t0
    timed("time_to_first_token", first_token or 0.0)
    timed("full_response", total)
    if tokens and total:
        print(f"  {'throughput':<44} {tokens / total:>10.1f} tok/s")
        RESULTS["tokens_per_second"] = round(tokens / total, 1)
    await provider.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write results as JSON")
    args = parser.parse_args()

    print(f"localcode benchmarks — Python {sys.version.split()[0]}")
    bench_indexing()
    asyncio.run(bench_tools())
    asyncio.run(bench_agent_harness())
    asyncio.run(bench_real_backend())
    if args.json:
        args.json.write_text(json.dumps(RESULTS, indent=2))
        print(f"\nresults written to {args.json}")


if __name__ == "__main__":
    main()
