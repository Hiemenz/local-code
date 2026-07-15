"""spawn_agent tool: run a sub-task in a fresh full-auto agent instance."""

from __future__ import annotations

from typing import Any

from localcode.tools.base import Tool, ToolResult


class SpawnAgentTool(Tool):
    name = "spawn_agent"
    description = (
        "Run an independent sub-task in a fresh full-auto agent and return its text output. "
        "Use for work that can be done in parallel or delegated to a clean context — "
        "e.g. 'refactor module X', 'write tests for Y', 'summarise file Z'. "
        "The sub-agent shares the same working directory and tools but has no memory "
        "of the current conversation."
    )
    permission = "execute"
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The complete task description for the sub-agent.",
            },
            "working_dir": {
                "type": "string",
                "description": "Subdirectory to restrict the sub-agent to (default: project root).",
            },
        },
        "required": ["prompt"],
    }

    async def run(self, prompt: str, working_dir: str | None = None, **_: Any) -> ToolResult:
        from localcode.agent.loop import Agent
        from localcode.agent.prompts import build_system_prompt
        from localcode.providers.base import create_provider
        from localcode.tools.base import ToolContext, build_default_registry

        root = self.ctx.resolve(working_dir) if working_dir else self.ctx.root
        provider = create_provider(self.ctx.config.provider)
        sub_ctx = ToolContext(root=root, config=self.ctx.config)
        sub_registry = build_default_registry(sub_ctx)
        # prevent infinite recursion
        sub_registry._tools.pop("spawn_agent", None)  # type: ignore[attr-defined]

        agent = Agent(
            provider=provider,
            registry=sub_registry,
            tool_ctx=sub_ctx,
            config=self.ctx.config,
        )
        agent.set_system_prompt(build_system_prompt(root, None, None))
        agent.mode = "full-auto"

        parts: list[str] = []
        try:
            async for event in agent.run_turn(prompt):
                if event.kind == "text" and event.text:
                    parts.append(event.text)
        finally:
            await provider.close()

        output = "".join(parts).strip() or "(sub-agent produced no text output)"
        return ToolResult(ok=True, output=output)


TOOLS = [SpawnAgentTool]
