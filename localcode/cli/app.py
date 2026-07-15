"""localcode CLI: interactive REPL and one-shot (-p) modes."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import mimetypes
import os
import re
import signal
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

from localcode import __version__
from localcode.agent.checkpoints import CheckpointManager
from localcode.agent.hooks import HookRunner
from localcode.agent.loop import Agent
from localcode.agent.notify import notify
from localcode.agent.prompts import build_system_prompt, load_project_instructions
from localcode.cli.approval import ConsoleApprover
from localcode.cli.render import StreamRenderer
from localcode.config.loader import cache_dir, data_dir, load_config
from localcode.config.schema import Config
from localcode.errors import ConfigError, LocalCodeError, ProviderError
from localcode.index.repomap import RepoIndex
from localcode.logging import get_logger, setup_logging
from localcode.providers.base import ImageContent, Provider, create_provider
from localcode.session.store import Session, SessionStore
from localcode.tools.base import ToolContext, build_default_registry

log = get_logger("cli")

_AT_FILE_RE = re.compile(r"@([^\s]+)")
_AT_FILE_MAX_CHARS = 50_000
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _expand_at_refs(line: str, root: Path) -> tuple[str, list[ImageContent]]:
    """Replace @path tokens:
    - Image files are base64-encoded and returned separately as ImageContent.
    - Text files are inlined as fenced code blocks.
    """
    images: list[ImageContent] = []

    def _replace(m: re.Match[str]) -> str:
        raw = m.group(1)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            candidate = candidate.resolve()
            if candidate.is_file():
                if candidate.suffix.lower() in _IMAGE_EXTS:
                    mime = mimetypes.guess_type(str(candidate))[0] or "image/png"
                    data = base64.standard_b64encode(candidate.read_bytes()).decode()
                    images.append(ImageContent(data=data, mime_type=mime))
                    return f"[Image: {raw}]"
                text = candidate.read_text(encoding="utf-8", errors="replace")
                if len(text) > _AT_FILE_MAX_CHARS:
                    text = text[:_AT_FILE_MAX_CHARS] + "\n... (truncated)"
                return f"\n[File: {raw}]\n```\n{text}\n```\n"
        except (OSError, ValueError):
            pass
        return m.group(0)

    text = _AT_FILE_RE.sub(_replace, line)
    return text, images


def _expand_at_files(line: str, root: Path) -> str:
    """Backward-compat wrapper: expand @file tokens, return text only (no images)."""
    text, _ = _expand_at_refs(line, root)
    return text


def _load_image(path: str) -> ImageContent:
    p = Path(path).resolve()
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    data = base64.standard_b64encode(p.read_bytes()).decode()
    return ImageContent(data=data, mime_type=mime)


class App:
    def __init__(self, root: Path, config: Config) -> None:
        self.root = root
        self.config = config
        self.console = Console()
        self.store = SessionStore(data_dir(), root)
        self.session: Session = self.store.create()
        self.checkpoints: CheckpointManager | None = CheckpointManager(
            data_dir() / "checkpoints" / self.session.meta.id
        )
        self.index: RepoIndex | None = None
        if config.index.enabled:
            cdir = config.cache.directory or cache_dir()
            self.index = RepoIndex(root, config.index, cdir if config.cache.enabled else None)
        self.tool_ctx = ToolContext(
            root=root, config=config, checkpoints=self.checkpoints, index=self.index
        )
        self.registry = build_default_registry(self.tool_ctx)
        self.provider: Provider = create_provider(config.provider)
        hooks = HookRunner(config.hooks) if config.hooks else None
        self.agent = Agent(
            provider=self.provider,
            registry=self.registry,
            tool_ctx=self.tool_ctx,
            config=config,
            session=self.session,
            approval=ConsoleApprover(self.console),
            hooks=hooks,
        )
        # Optional worktree manager — set by CLI when --worktree is passed
        self.worktree: object | None = None

    # -- lifecycle -------------------------------------------------------------

    async def startup(self) -> None:
        if self.index is not None:
            with self.console.status("Indexing repository…"):
                await asyncio.to_thread(self.index.scan)
        from localcode.mcp.client import load_mcp_tools
        await load_mcp_tools(self.registry, self.tool_ctx, self.root)
        self.rebuild_system_prompt()
        instructions = load_project_instructions(self.root)
        if instructions is not None:
            name, _ = instructions
            self.console.print(f"[dim]Project instructions loaded from {name}.[/]")
        if self.agent.hooks:
            await self.agent.hooks.fire("session_start")

    def rebuild_system_prompt(self) -> None:
        repo_map = None
        summary = None
        if self.index is not None:
            repo_map = self.index.render_map(self.config.context.repo_map_tokens * 4)
            summary = self.index.project.summary()
        self.agent.set_system_prompt(
            build_system_prompt(
                self.root,
                repo_map,
                summary,
                plan_mode=self.agent.mode == "plan",
                instructions=load_project_instructions(self.root),
            )
        )

    def attach_session(self, session: Session) -> None:
        self.session = session
        self.agent.session = session
        self.agent.messages = session.to_messages()
        self.checkpoints = CheckpointManager(data_dir() / "checkpoints" / session.meta.id)
        self.tool_ctx.checkpoints = self.checkpoints

    async def switch_model(self, name: str) -> None:
        if name in self.config.providers:
            new_cfg = self.config.providers[name]
            await self.provider.close()
            self.config.provider = new_cfg
            self.provider = create_provider(new_cfg)
            self.agent.provider = self.provider
            self.agent.context_mgr.context_length = new_cfg.context_length
            self.console.print(f"Switched to provider [bold]{name}[/] ({new_cfg.model})")
        else:
            self.config.provider.model = name
            self.console.print(f"Model set to [bold]{name}[/]")

    def _ctx_toolbar(self) -> HTML:
        """Live prompt bottom-toolbar: context bar + mode + session id."""
        ctx = self.agent.context_mgr.total_tokens(self.agent._full_messages())
        total = self.config.provider.context_length
        pct = ctx / total if total else 0
        k_used = ctx // 1000
        k_total = total // 1000
        filled = min(round(pct * 16), 16)
        bar = "▓" * filled + "░" * (16 - filled)
        if pct > 0.85:
            bar_html = f"<ansired>[{bar}]</ansired>"
        elif pct > 0.70:
            bar_html = f"<ansiyellow>[{bar}]</ansiyellow>"
        else:
            bar_html = f"[{bar}]"
        return HTML(
            f" ctx {k_used}k/{k_total}k {bar_html}"
            f"  mode: {self.agent.mode}"
            f"  {self.session.meta.id[:22]}"
        )

    # -- run modes ----------------------------------------------------------------

    async def run_turn_interruptible(
        self,
        text: str,
        renderer: StreamRenderer,
        images: list[ImageContent] | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert task is not None

        def on_sigint() -> None:
            self.agent.interrupt()
            task.cancel()

        loop.add_signal_handler(signal.SIGINT, on_sigint)
        files_modified: list[str] = []
        try:
            async for event in self.agent.run_turn(text, images=images):
                renderer.render_event(event)
                if event.kind == "turn_end":
                    files_modified = event.stats.get("files_modified", [])  # type: ignore[assignment]
        except asyncio.CancelledError:
            renderer.flush()
            self.console.print("[yellow]Interrupted — state saved; continue typing.[/]")
        except ProviderError as e:
            renderer.flush()
            self.console.print(f"[red]Provider error:[/] {e}")
        finally:
            renderer.flush()
            loop.remove_signal_handler(signal.SIGINT)

        if self.config.notifications.enabled and files_modified:
            n = len(files_modified)
            body = f"{n} file(s) modified: {', '.join(files_modified[:3])}"
            if n > 3:
                body += f" +{n - 3} more"
            asyncio.create_task(
                notify(
                    "localcode — turn complete",
                    body,
                    backend=self.config.notifications.backend,
                    ntfy_topic=self.config.notifications.ntfy_topic,
                )
            )

    async def repl(self) -> None:
        await self.startup()
        self._banner()
        history = FileHistory(str(data_dir() / "input_history"))
        bindings = KeyBindings()

        @bindings.add("enter")
        def _accept(event: object) -> None:  # Enter submits
            event.current_buffer.validate_and_handle()  # type: ignore[attr-defined]

        @bindings.add("escape", "enter")
        def _newline(event: object) -> None:  # Alt+Enter inserts a newline
            event.current_buffer.insert_text("\n")  # type: ignore[attr-defined]

        prompt: PromptSession[str] = PromptSession(
            history=history, multiline=True, key_bindings=bindings,
            bottom_toolbar=self._ctx_toolbar,
        )
        renderer = StreamRenderer(self.console, context_length=self.config.provider.context_length)
        from localcode.cli.commands import dispatch

        while True:
            try:
                line = (await prompt.prompt_async("\n› ")).strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            if not line:
                continue
            if line.startswith("/"):
                if not await dispatch(self, line):
                    break
                continue
            if line.startswith("!"):
                await self._passthrough(line[1:].strip())
                continue
            line, at_images = _expand_at_refs(line, self.root)
            await self.run_turn_interruptible(line, renderer, images=at_images or None)
        if self.agent.hooks:
            await self.agent.hooks.fire("session_end")
        await self.provider.close()
        self.console.print("[dim]Bye.[/]")

    async def _passthrough(self, command: str) -> None:
        from localcode.providers.base import Message
        from localcode.tools.shell import run_subprocess

        result = await run_subprocess(command, str(self.root), 300, 100_000)
        self.console.print(result.output)
        self.agent.messages.append(
            Message(role="user", content=f"$ {command}\n{result.output[:4000]}")
        )

    async def run_once(
        self, text: str, images: list[ImageContent] | None = None
    ) -> int:
        """Non-interactive: run one turn, print plain text, exit."""
        await self.startup()
        self.agent.approval = None  # no interactive gate in -p mode
        exit_code = 0
        try:
            async for event in self.agent.run_turn(text, images=images):
                if event.kind == "text":
                    sys.stdout.write(event.text)
                    sys.stdout.flush()
                elif event.kind == "tool_start" and event.tool_call is not None:
                    from localcode.agent.loop import format_call

                    print(f"\n[tool] {format_call(event.tool_call)}", file=sys.stderr)
        except LocalCodeError as e:
            print(f"error: {e}", file=sys.stderr)
            exit_code = 1
        print()
        if self.agent.hooks:
            await self.agent.hooks.fire("session_end")
        await self.provider.close()
        return exit_code

    def _banner(self) -> None:
        indexed = f"{len(self.index.files)} files indexed" if self.index else "index off"
        ctx_k = self.config.provider.context_length // 1000
        self.console.print(
            f"[bold cyan]localcode[/] v{__version__} · [bold]{self.config.provider.model}[/] "
            f"via {self.config.provider.type} @ {self.config.provider.base_url}\n"
            f"[dim]{self.root} · {indexed} · mode: {self.agent.mode} · "
            f"ctx: {ctx_k}k · session: {self.session.meta.id}[/]\n"
            f"[dim]/help for commands · Ctrl+C interrupts · Ctrl+D exits · "
            f"@file expands file contents[/]"
        )


def _apply_env_overrides(config: Config) -> None:
    if model := os.environ.get("LOCALCODE_MODEL"):
        config.provider.model = model
    if base := os.environ.get("LOCALCODE_BASE_URL"):
        config.provider.base_url = base
    if ptype := os.environ.get("LOCALCODE_PROVIDER"):
        config.provider.type = ptype  # type: ignore[assignment]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localcode",
        description="A Claude Code-compatible coding assistant on local LLMs",
    )
    parser.add_argument("path", nargs="?", default=".", help="project directory")
    parser.add_argument("-p", "--print", dest="oneshot", metavar="PROMPT",
                        help="run one prompt non-interactively and exit")
    parser.add_argument("--image", dest="images", metavar="FILE", action="append",
                        default=[],
                        help="attach an image to the prompt (use with -p; repeatable)")
    parser.add_argument("--config", type=Path, help="extra config file")
    parser.add_argument("--model", help="override model name")
    parser.add_argument("--mode", choices=["plan", "ask", "auto-edit", "full-auto"],
                        help="approval mode")
    parser.add_argument("--resume", nargs="?", const="latest", metavar="ID",
                        help="resume a session (latest if no id)")
    parser.add_argument("--continue", dest="cont", action="store_true",
                        help="continue the most recent session")
    parser.add_argument("--worktree", action="store_true",
                        help="run the agent in an isolated git worktree")
    parser.add_argument("--serve", action="store_true", help="start the HTTP API server")
    parser.add_argument("--shell-integration", dest="shell_integration", action="store_true",
                        help="print zsh shell integration script and exit")
    parser.add_argument("--version", action="version", version=f"localcode {__version__}")
    return parser


def main() -> None:
    import atexit

    args = build_parser().parse_args()

    if args.shell_integration:
        from localcode.shell.integration import zsh_script
        print(zsh_script(), end="")
        return

    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        sys.exit(2)
    try:
        config = load_config(project_root=root, explicit=args.config)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    _apply_env_overrides(config)
    if args.model:
        config.provider.model = args.model
    if args.mode:
        config.permissions.approval_mode = args.mode

    setup_logging(
        config.logging.directory or (data_dir() / "logs"),
        config.logging.level,
        config.logging.console,
    )

    app = App(root, config)
    if args.resume or args.cont:
        session = (
            app.store.latest()
            if args.cont or args.resume == "latest"
            else app.store.load(args.resume)
        )
        if session is not None:
            app.attach_session(session)

    from localcode.shell.integration import clear_active, write_active
    _proj = str(root)
    write_active(data_dir(), _proj, app.session.meta.id)
    atexit.register(clear_active, data_dir(), _proj)

    # Load --image attachments for -p mode
    cli_images: list[ImageContent] = []
    for img_path in args.images:
        try:
            cli_images.append(_load_image(img_path))
        except OSError as e:
            print(f"error loading image {img_path}: {e}", file=sys.stderr)
            sys.exit(2)

    try:
        if args.serve:
            from localcode.api.server import serve

            asyncio.run(app.startup())
            serve(app.agent, app.store, config.api.host, config.api.port)
        elif args.oneshot:
            sys.exit(asyncio.run(app.run_once(args.oneshot, images=cli_images or None)))
        else:
            if args.worktree:
                asyncio.run(_setup_worktree(app))
            with contextlib.suppress(KeyboardInterrupt):
                asyncio.run(app.repl())
    except LocalCodeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


async def _setup_worktree(app: App) -> None:
    """Create a git worktree and redirect the agent's workspace into it."""
    from localcode.agent.worktree import WorktreeManager

    mgr = WorktreeManager(app.root, app.session.meta.id, data_dir())
    try:
        wt_path = await mgr.create()
    except RuntimeError as e:
        app.console.print(f"[red]Worktree setup failed:[/] {e}")
        return
    app.worktree = mgr
    app.tool_ctx.root = wt_path
    app.console.print(
        f"[dim]Worktree active at {wt_path}. "
        "Use /worktree merge or /worktree discard when done.[/]"
    )


if __name__ == "__main__":
    main()
