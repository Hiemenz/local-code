import sys
from typing import Any

from localcode.agent.loop import Agent, AgentEvent
from localcode.config.schema import Config, IndexConfig
from localcode.index.repomap import RepoIndex
from localcode.providers.base import Message, ToolCall
from localcode.tools.base import Tool, ToolContext, build_default_registry


def make_agent(
    tool_ctx: ToolContext,
    provider: Any,
    config: Config,
    approval: Any = None,
    initial: list[Message] | None = None,
) -> Agent:
    registry = build_default_registry(tool_ctx)
    agent = Agent(
        provider=provider,
        registry=registry,
        tool_ctx=tool_ctx,
        config=config,
        approval=approval,
        initial_messages=initial,
    )
    agent.set_system_prompt("You are a test agent.")
    return agent


async def collect(agent: Agent, text: str) -> list[AgentEvent]:
    return [event async for event in agent.run_turn(text)]


async def test_plain_text_turn(tool_ctx, config, fake_provider_factory) -> None:
    config.agent.auto_test = False
    provider = fake_provider_factory([("Hello there!", [])])
    agent = make_agent(tool_ctx, provider, config)
    events = await collect(agent, "hi")
    text = "".join(e.text for e in events if e.kind == "text")
    assert text == "Hello there!"
    assert [m.role for m in agent.messages] == ["user", "assistant"]


async def test_tool_call_roundtrip(tool_ctx, config, fake_provider_factory) -> None:
    config.agent.auto_test = False
    provider = fake_provider_factory(
        [
            ("", [ToolCall(id="1", name="read_file", arguments={"path": "src/util.py"})]),
            ("The helper returns 42.", []),
        ]
    )
    agent = make_agent(tool_ctx, provider, config)
    events = await collect(agent, "what does helper return?")
    ends = [e for e in events if e.kind == "tool_end"]
    assert len(ends) == 1 and ends[0].result is not None and ends[0].result.ok
    assert "return 42" in ends[0].result.output
    # tool result was fed back to the model
    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert tool_msgs and tool_msgs[0].tool_call_id == "1"
    assert "The helper returns 42." in agent.messages[-1].content


async def test_plan_mode_blocks_writes(tool_ctx, config, fake_provider_factory) -> None:
    config.agent.auto_test = False
    provider = fake_provider_factory(
        [
            ("", [ToolCall(id="1", name="write_file", arguments={"path": "x.py", "content": "x"})]),
            ("ok", []),
        ]
    )
    agent = make_agent(tool_ctx, provider, config)
    agent.mode = "plan"
    events = await collect(agent, "change something")
    ends = [e for e in events if e.kind == "tool_end"]
    assert ends and not ends[0].result.ok
    assert "Plan mode" in ends[0].result.output
    assert not (tool_ctx.root / "x.py").exists()
    # plan mode only advertises read-only tools
    assert all(
        agent.registry.get(spec.name).permission == "read" for spec in agent._tool_specs()
    )


async def test_approval_deny_and_always(tool_ctx, config, fake_provider_factory) -> None:
    config.agent.auto_test = False
    decisions = ["deny", "always"]
    prompts: list[str] = []

    async def approver(tool: Tool, args: dict) -> str:
        prompts.append(tool.name)
        return decisions.pop(0)

    provider = fake_provider_factory(
        [
            ("", [ToolCall(id="1", name="bash", arguments={"command": "touch made1"})]),
            ("", [ToolCall(id="2", name="bash", arguments={"command": "touch made2"})]),
            ("", [ToolCall(id="3", name="bash", arguments={"command": "touch made3"})]),
            ("done", []),
        ]
    )
    agent = make_agent(tool_ctx, provider, config, approval=approver)
    events = await collect(agent, "make files")
    ends = [e for e in events if e.kind == "tool_end"]
    assert not ends[0].result.ok and "denied" in ends[0].result.output.lower()
    assert not (tool_ctx.root / "made1").exists()
    assert ends[1].result.ok and (tool_ctx.root / "made2").exists()
    # 'always' covered the third call: no new prompt
    assert ends[2].result.ok and (tool_ctx.root / "made3").exists()
    assert prompts == ["bash", "bash"]


async def test_deny_list_blocks_even_full_auto(tool_ctx, config, fake_provider_factory) -> None:
    config.agent.auto_test = False
    provider = fake_provider_factory(
        [("", [ToolCall(id="1", name="bash", arguments={"command": "sudo rm -rf /x"})]), ("ok", [])]
    )
    agent = make_agent(tool_ctx, provider, config)
    agent.mode = "full-auto"
    events = await collect(agent, "clean up")
    ends = [e for e in events if e.kind == "tool_end"]
    assert not ends[0].result.ok
    assert "deny list" in ends[0].result.output


async def test_unknown_tool_reports_available(tool_ctx, config, fake_provider_factory) -> None:
    config.agent.auto_test = False
    provider = fake_provider_factory(
        [("", [ToolCall(id="1", name="teleport", arguments={})]), ("ok", [])]
    )
    agent = make_agent(tool_ctx, provider, config)
    events = await collect(agent, "go")
    ends = [e for e in events if e.kind == "tool_end"]
    assert not ends[0].result.ok
    assert "read_file" in ends[0].result.output  # lists available tools


async def test_edit_verification_catches_missing_file(tool_ctx, config, fake_provider_factory) -> None:
    config.agent.auto_test = False
    config.agent.verify_edits = True
    provider = fake_provider_factory(
        [
            ("", [ToolCall(id="1", name="write_file",
                           arguments={"path": "out.txt", "content": "hello"})]),
            ("done", []),
        ]
    )
    agent = make_agent(tool_ctx, provider, config)
    agent.mode = "full-auto"
    events = await collect(agent, "write it")
    ends = [e for e in events if e.kind == "tool_end"]
    assert ends[0].result.ok  # file exists, verification passed


async def test_auto_test_fix_cycle(tool_ctx, config, fake_provider_factory) -> None:
    config.agent.auto_test = True
    config.agent.fix_iterations = 2
    index = RepoIndex(tool_ctx.root, IndexConfig(), None)
    index.scan()
    check = (
        f"{sys.executable} -c \"import sys; "
        f"sys.exit(0 if open('flag.txt').read().strip() == 'ok' else 1)\""
    )
    index.project.test_command = check
    tool_ctx.index = index

    provider = fake_provider_factory(
        [
            ("", [ToolCall(id="1", name="write_file",
                           arguments={"path": "flag.txt", "content": "bad"})]),
            ("I wrote the flag.", []),  # -> auto-test fails, failure pushed back
            ("", [ToolCall(id="2", name="write_file",
                           arguments={"path": "flag.txt", "content": "ok"})]),
            ("Fixed the flag.", []),  # -> auto-test passes
        ]
    )
    agent = make_agent(tool_ctx, provider, config)
    agent.mode = "full-auto"
    events = await collect(agent, "write flag.txt")
    infos = [e.text for e in events if e.kind == "info"]
    assert any("Tests failed" in t for t in infos)
    assert any("Tests passed" in t for t in infos)
    assert (tool_ctx.root / "flag.txt").read_text() == "ok"
    # the failure round-trip appended a user message with the test output
    assert any("test suite failed" in m.content.lower() for m in agent.messages if m.role == "user")


async def test_compaction_triggered_in_turn(tool_ctx, config, fake_provider_factory) -> None:
    config.agent.auto_test = False
    config.provider.context_length = 300
    config.context.compact_threshold = 0.5
    config.context.keep_recent_messages = 2
    initial = [
        Message(role="user", content="old question " + "x" * 400),
        Message(role="assistant", content="old answer " + "y" * 400),
        Message(role="user", content="another " + "z" * 400),
        Message(role="assistant", content="fine"),
    ]
    provider = fake_provider_factory([("SUMMARY", []), ("new reply", [])])
    agent = make_agent(tool_ctx, provider, config, initial=initial)
    events = await collect(agent, "continue")
    infos = [e.text for e in events if e.kind == "info"]
    assert any("compact" in t.lower() for t in infos)
    assert any("[Conversation summary" in m.content for m in agent.messages)


async def test_turn_end_stats(tool_ctx, config, fake_provider_factory) -> None:
    config.agent.auto_test = False
    provider = fake_provider_factory([("hi", [])])
    agent = make_agent(tool_ctx, provider, config)
    events = await collect(agent, "hello")
    end = events[-1]
    assert end.kind == "turn_end"
    assert end.stats["prompt_tokens"] == 10
    assert end.stats["completion_tokens"] == 5
