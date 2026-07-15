"""The agentic loop: stream → tool calls → results → repeat, with approval
gates, parallel read-only execution, edit verification, auto-test-and-fix,
retries, and context compaction."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from localcode.agent.hooks import HookRunner, make_result_env, make_tool_env
from localcode.config.schema import ApprovalMode, Config
from localcode.context.manager import ContextManager
from localcode.errors import ProviderError, ToolError
from localcode.logging import get_logger
from localcode.providers.base import ImageContent, Message, Provider, ToolCall, Usage
from localcode.session.store import Session
from localcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, truncate

log = get_logger("agent")

Decision = Literal["allow", "always", "deny"]
ApprovalCallback = Callable[[Tool, dict[str, Any]], Awaitable[Decision]]


@dataclass
class AgentEvent:
    kind: Literal["text", "tool_start", "tool_end", "info", "turn_end"]
    text: str = ""
    tool_call: ToolCall | None = None
    result: ToolResult | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        tool_ctx: ToolContext,
        config: Config,
        session: Session | None = None,
        approval: ApprovalCallback | None = None,
        initial_messages: list[Message] | None = None,
        hooks: HookRunner | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.tool_ctx = tool_ctx
        self.config = config
        self.session = session
        self.approval = approval
        self.hooks = hooks
        self.mode: ApprovalMode = config.permissions.approval_mode
        self.messages: list[Message] = initial_messages or []
        self.context_mgr = ContextManager(config.context, config.provider.context_length)
        self.total_usage = Usage()
        self._system_prompt: str = ""
        self._session_allowed: set[str] = set()
        self._interrupted = False

    # -- setup ---------------------------------------------------------------

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def interrupt(self) -> None:
        self._interrupted = True

    def _tool_specs(self) -> list[Any]:
        if self.mode == "plan":
            return [t.spec() for t in self.registry if t.permission == "read"]
        return self.registry.specs()

    def _full_messages(self) -> list[Message]:
        return [Message(role="system", content=self._system_prompt), *self.messages]

    # -- approval --------------------------------------------------------------

    def _is_denied(self, tool: Tool, args: dict[str, Any]) -> bool:
        if tool.name == "bash":
            command = str(args.get("command", "")).strip()
            return any(command.startswith(d) for d in self.config.permissions.deny_commands)
        return False

    def _needs_approval(self, tool: Tool, args: dict[str, Any]) -> bool:
        if tool.permission == "read":
            return False
        if tool.name in self.config.permissions.allow_tools or tool.name in self._session_allowed:
            return False
        if tool.name == "bash":
            command = str(args.get("command", "")).strip()
            if any(
                command == a or command.startswith(a + " ")
                for a in [*self.config.permissions.allow_commands, *self._session_allowed]
            ):
                return False
        if self.mode == "full-auto":
            return False
        if self.mode == "auto-edit":
            return tool.permission == "execute"
        return True  # ask mode

    async def _gate(self, tool: Tool, args: dict[str, Any]) -> ToolResult | None:
        """Return a denial ToolResult, or None when execution may proceed."""
        if self._is_denied(tool, args):
            return ToolResult.error("This command is on the deny list and was blocked.")
        if self.mode == "plan" and tool.permission != "read":
            return ToolResult.error("Plan mode is active: only read-only tools are allowed.")
        if not self._needs_approval(tool, args):
            return None
        if self.approval is None:
            return ToolResult.error("Approval required but no approver available; action skipped.")
        decision = await self.approval(tool, args)
        if decision == "deny":
            return ToolResult.error(
                "User denied this action. Ask for guidance or try another approach."
            )
        if decision == "always":
            if tool.name == "bash":
                command = str(args.get("command", "")).split("\n")[0]
                # allow the program name for the rest of the session ("git", "touch"…)
                prefix = command.split()[0] if command.split() else command
                self._session_allowed.add(prefix)
            else:
                self._session_allowed.add(tool.name)
        return None

    # -- execution ---------------------------------------------------------------

    async def _execute_one(self, call: ToolCall) -> ToolResult:
        tool = self.registry.get(call.name)
        if tool is None:
            known = ", ".join(self.registry.names())
            return ToolResult.error(f"Unknown tool '{call.name}'. Available tools: {known}")
        denial = await self._gate(tool, call.arguments)
        if denial is not None:
            return denial
        if self.hooks:
            await self.hooks.fire(
                "pre_tool",
                tool_name=call.name,
                extra_env=make_tool_env(call.arguments),
            )
        try:
            result = await tool.run(**call.arguments)
        except TypeError as e:
            return ToolResult.error(f"Invalid arguments for {call.name}: {e}")
        except ToolError as e:
            return ToolResult.error(str(e))
        except Exception as e:  # noqa: BLE001 - tool bugs must not kill the loop
            log.exception(f"tool {call.name} crashed")
            return ToolResult.error(f"{call.name} crashed: {e}")
        if (
            self.config.agent.verify_edits
            and result.ok
            and tool.permission == "write"
            and "path" in call.arguments
        ):
            result = self._verify_edit(call, result)
        if self.hooks:
            await self.hooks.fire(
                "post_tool",
                tool_name=call.name,
                extra_env=make_result_env(result.ok, result.output),
            )
        return result

    def _verify_edit(self, call: ToolCall, result: ToolResult) -> ToolResult:
        try:
            target = self.tool_ctx.resolve(str(call.arguments["path"]))
        except Exception:  # noqa: BLE001
            return result
        if call.name in ("write_file", "edit_file") and not target.is_file():
            return ToolResult.error(f"Verification failed: {target} does not exist after edit")
        if call.name == "edit_file":
            new_string = str(call.arguments.get("new_string", ""))
            if new_string and new_string not in target.read_text(encoding="utf-8"):
                return ToolResult.error(
                    "Verification failed: new_string not present in file after edit"
                )
        return result

    async def _execute_calls(
        self, calls: list[ToolCall]
    ) -> AsyncIterator[tuple[ToolCall, ToolResult]]:
        """Run read-only calls concurrently, mutating calls sequentially, and
        yield results in the original order so the transcript stays aligned."""

        def is_readonly(call: ToolCall) -> bool:
            tool = self.registry.get(call.name)
            return tool is not None and tool.permission == "read"

        if self.config.agent.parallel_tools and len(calls) > 1 and all(map(is_readonly, calls)):
            results = await asyncio.gather(*(self._execute_one(c) for c in calls))
            for call, result in zip(calls, results, strict=True):
                yield call, result
        else:
            for call in calls:
                yield call, await self._execute_one(call)

    # -- provider ---------------------------------------------------------------

    async def _stream_model(self, out: dict[str, Any]) -> AsyncIterator[AgentEvent]:
        """One model call with retries. Yields live text deltas; the collected
        text and tool calls are placed in ``out``. Only errors that occur
        before any text was emitted are retried (nothing was shown yet)."""
        attempt = 0
        while True:
            text_parts: list[str] = []
            calls: list[ToolCall] = []
            try:
                async for event in self.provider.chat(self._full_messages(), self._tool_specs()):
                    if event.kind == "text":
                        text_parts.append(event.text)
                        yield AgentEvent(kind="text", text=event.text)
                    elif event.kind == "tool_call" and event.tool_call is not None:
                        calls.append(event.tool_call)
                    elif event.kind == "usage" and event.usage is not None:
                        self.total_usage.prompt_tokens += event.usage.prompt_tokens
                        self.total_usage.completion_tokens += event.usage.completion_tokens
                out["text"] = "".join(text_parts)
                out["calls"] = calls
                return
            except ProviderError as e:
                attempt += 1
                if text_parts or not e.retryable or attempt > self.config.agent.max_retries:
                    raise
                delay = self.config.agent.retry_base_delay * (2 ** (attempt - 1))
                log.info(f"provider retry {attempt} in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)

    # -- the main loop -------------------------------------------------------------

    async def run_turn(
        self,
        user_input: str,
        images: list[ImageContent] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        self._interrupted = False
        self.tool_ctx.files_modified.clear()
        if self.tool_ctx.checkpoints is not None:
            self.tool_ctx.checkpoints.begin_turn(user_input)
        self.messages.append(
            Message(role="user", content=user_input, images=images or [])
        )
        self._record({"type": "user", "content": user_input})

        fix_attempts = 0
        tests_ran = False
        try:
            for _round in range(self.config.agent.max_tool_rounds):
                if self.context_mgr.needs_compaction(self.messages):
                    async for event in self._compact():
                        yield event

                out: dict[str, Any] = {}
                async for event in self._stream_model(out):
                    yield event
                text: str = out.get("text", "")
                calls: list[ToolCall] = out.get("calls", [])
                self.messages.append(Message(role="assistant", content=text, tool_calls=calls))
                self._record(
                    {
                        "type": "assistant",
                        "content": text,
                        "tool_calls": [
                            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls
                        ],
                    }
                )

                if not calls:
                    # Model is done. Optionally verify with the test suite.
                    if self._should_auto_test(tests_ran, fix_attempts):
                        tests_ran = True
                        failed_output = await self._run_auto_tests()
                        if failed_output is not None:
                            fix_attempts += 1
                            yield AgentEvent(
                                kind="info",
                                text=f"Tests failed — asking the model to fix (attempt {fix_attempts})",
                            )
                            self._push_test_failure(failed_output)
                            tests_ran = False
                            continue
                        yield AgentEvent(kind="info", text="Tests passed.")
                    break

                for call in calls:
                    yield AgentEvent(kind="tool_start", tool_call=call)
                async for call, result in self._execute_calls(calls):
                    if self._interrupted:
                        result = ToolResult.error("Interrupted by user.")
                    yield AgentEvent(kind="tool_end", tool_call=call, result=result)
                    content = truncate(result.output, self.tool_ctx.max_output_chars())
                    if not result.ok and not content.startswith("Error"):
                        content = f"Error: {content}"
                    self.messages.append(
                        Message(role="tool", content=content, tool_call_id=call.id, name=call.name)
                    )
                    self._record(
                        {
                            "type": "tool_result",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": content,
                            "ok": result.ok,
                        }
                    )
                if self._interrupted:
                    yield AgentEvent(kind="info", text="Turn interrupted.")
                    break
            else:
                yield AgentEvent(
                    kind="info",
                    text=f"Stopped after {self.config.agent.max_tool_rounds} tool rounds.",
                )
        finally:
            if self.tool_ctx.checkpoints is not None:
                self.tool_ctx.checkpoints.end_turn()
            if self.tool_ctx.index is not None and self.tool_ctx.files_modified:
                await asyncio.to_thread(self.tool_ctx.index.refresh)

        stats = {
            "prompt_tokens": self.total_usage.prompt_tokens,
            "completion_tokens": self.total_usage.completion_tokens,
            "context_tokens": self.context_mgr.total_tokens(self._full_messages()),
            "files_modified": sorted(str(p) for p in self.tool_ctx.files_modified),
        }
        if self.hooks:
            await self.hooks.fire(
                "turn_end",
                extra_env={
                    "LOCALCODE_FILES_MODIFIED": ",".join(stats["files_modified"]),  # type: ignore[arg-type]
                },
            )
        yield AgentEvent(kind="turn_end", stats=stats)

    # -- auto-test ----------------------------------------------------------------

    def _should_auto_test(self, tests_ran: bool, fix_attempts: int) -> bool:
        return (
            self.config.agent.auto_test
            and self.mode != "plan"
            and not tests_ran
            and bool(self.tool_ctx.files_modified)
            and fix_attempts < self.config.agent.fix_iterations
            and self._test_command() is not None
        )

    def _test_command(self) -> str | None:
        if self.tool_ctx.index is not None:
            return self.tool_ctx.index.project.test_command
        return None

    async def _run_auto_tests(self) -> str | None:
        """Run the project's tests. Returns failure output, or None if green."""
        from localcode.tools.shell import run_subprocess

        command = self._test_command()
        assert command is not None
        result = await run_subprocess(
            command, str(self.tool_ctx.root), 600, self.tool_ctx.max_output_chars()
        )
        return None if result.ok else result.output

    def _push_test_failure(self, output: str) -> None:
        content = (
            "The test suite failed after your changes. Fix the failures, then stop.\n"
            f"Test output:\n{output}"
        )
        self.messages.append(Message(role="user", content=content))
        self._record({"type": "user", "content": content})

    # -- compaction -----------------------------------------------------------------

    async def _compact(self) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(kind="info", text="Context is getting long — compacting…")
        self.messages = await self.context_mgr.compact(self.messages, self.provider)
        summary = next((m for m in self.messages if m.content.startswith("[Conversation summary")), None)
        if summary is not None:
            self._record({"type": "compaction", "content": summary.content})

    async def force_compact(self) -> int:
        """Manual /compact: summarize regardless of threshold. Returns new token count."""
        keep = self.context_mgr.cfg.keep_recent_messages
        self.context_mgr.cfg.keep_recent_messages = 2
        try:
            self.messages = await self.context_mgr.compact(self.messages, self.provider)
        finally:
            self.context_mgr.cfg.keep_recent_messages = keep
        return self.context_mgr.total_tokens(self._full_messages())

    def _record(self, event: dict[str, Any]) -> None:
        if self.session is not None:
            self.session.append(event)


def format_call(call: ToolCall) -> str:
    args = json.dumps(call.arguments, ensure_ascii=False, default=str)
    if len(args) > 120:
        args = args[:119] + "…"
    return f"{call.name}({args})"
