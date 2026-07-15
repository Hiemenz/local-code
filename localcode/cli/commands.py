"""Slash commands. Mirrors Claude Code's command set where it makes sense."""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from rich.table import Table

if TYPE_CHECKING:
    from localcode.cli.app import App

Handler = Callable[["App", str], Awaitable[bool]]  # returns False to exit the REPL

COMMANDS: dict[str, tuple[str, Handler]] = {}


def command(name: str, help_text: str) -> Callable[[Handler], Handler]:
    def wrap(fn: Handler) -> Handler:
        COMMANDS[name] = (help_text, fn)
        return fn
    return wrap


def register_command(name: str, help_text: str, handler: Handler) -> None:
    """Register a slash command from a plugin."""
    COMMANDS[name] = (help_text, handler)


async def dispatch(app: App, line: str) -> bool:
    name, _, args = line.lstrip("/").partition(" ")
    entry = COMMANDS.get(name)
    if entry is None:
        app.console.print(f"[red]Unknown command: /{name}[/] — try /help")
        return True
    return await entry[1](app, args.strip())


@command("help", "Show available commands")
async def _help(app: App, args: str) -> bool:
    table = Table(show_header=False, box=None, padding=(0, 2))
    for name, (help_text, _) in sorted(COMMANDS.items()):
        table.add_row(f"/{name}", help_text)
    app.console.print(table)
    app.console.print(
        "[dim]Also: '!cmd' runs a shell command directly; Ctrl+C interrupts; "
        "Ctrl+D or /quit exits.[/]"
    )
    return True


@command("clear", "Clear conversation history (keeps the session file)")
async def _clear(app: App, args: str) -> bool:
    app.agent.messages.clear()
    app.console.print("[dim]Conversation cleared.[/]")
    return True


@command("compact", "Summarize older history to free context")
async def _compact(app: App, args: str) -> bool:
    if not app.agent.messages:
        app.console.print("[dim]Nothing to compact.[/]")
        return True
    with app.console.status("Compacting…"):
        tokens = await app.agent.force_compact()
    app.console.print(f"[dim]Compacted. Context is now ~{tokens} tokens.[/]")
    return True


@command("cost", "Show token usage for this session")
async def _cost(app: App, args: str) -> bool:
    usage = app.agent.total_usage
    ctx = app.agent.context_mgr.total_tokens(app.agent._full_messages())
    app.console.print(
        f"prompt tokens: {usage.prompt_tokens}  ·  completion tokens: "
        f"{usage.completion_tokens}  ·  current context: ~{ctx} "
        f"/ {app.config.provider.context_length}"
    )
    return True


@command("mode", "Show or set approval mode: plan | ask | auto-edit | full-auto")
async def _mode(app: App, args: str) -> bool:
    if args in ("plan", "ask", "auto-edit", "full-auto"):
        app.agent.mode = args  # type: ignore[assignment]
        app.rebuild_system_prompt()
    elif args:
        app.console.print("[red]Modes: plan, ask, auto-edit, full-auto[/]")
    app.console.print(f"Approval mode: [bold]{app.agent.mode}[/]")
    return True


@command("plan", "Toggle plan mode (read-only investigation)")
async def _plan(app: App, args: str) -> bool:
    app.agent.mode = "ask" if app.agent.mode == "plan" else "plan"
    app.rebuild_system_prompt()
    state = "ON — read-only" if app.agent.mode == "plan" else "off"
    app.console.print(f"Plan mode {state}.")
    return True


@command("model", "Show/switch model: /model [pull <name> | <name> | provider-name]")
async def _model(app: App, args: str) -> bool:
    if args.startswith("pull ") or args == "pull":
        name = args[4:].strip()
        if not name:
            app.console.print("[red]Usage: /model pull <name>[/]")
            return True
        from localcode.providers.ollama import OllamaProvider
        if not isinstance(app.agent.provider, OllamaProvider):
            app.console.print(
                "[yellow]/model pull is only available for the ollama provider.[/]"
            )
            return True
        with app.console.status(f"Pulling {name}…") as spinner:
            try:
                async for status in app.agent.provider.pull_model(name):
                    spinner.update(f"[dim]{status}[/]")
            except Exception as e:  # noqa: BLE001
                app.console.print(f"[red]Pull failed:[/] {e}")
                return True
        app.console.print(f"[green]✓[/] {name} pulled. Use /model {name} to switch.")
        return True
    if not args:
        app.console.print(f"Model: [bold]{app.config.provider.model}[/] "
                          f"({app.config.provider.type} @ {app.config.provider.base_url})")
        try:
            models = await app.agent.provider.list_models()
            app.console.print("[dim]Available: " + ", ".join(models[:20]) + "[/]")
        except Exception as e:  # noqa: BLE001
            app.console.print(f"[dim]Could not list models: {e}[/]")
        return True
    await app.switch_model(args)
    return True


@command("sessions", "List recent sessions for this project")
async def _sessions(app: App, args: str) -> bool:
    metas = app.store.list_sessions()
    if not metas:
        app.console.print("[dim]No sessions yet.[/]")
        return True
    table = Table(box=None)
    table.add_column("id", style="cyan")
    table.add_column("title")
    for m in metas[:15]:
        table.add_row(m.id, m.title)
    app.console.print(table)
    app.console.print("[dim]Resume with: /resume <id>[/]")
    return True


@command("resume", "Resume a previous session: /resume [id] (default: latest)")
async def _resume(app: App, args: str) -> bool:
    session = app.store.load(args) if args else app.store.latest()
    if session is None:
        app.console.print("[red]Session not found.[/]")
        return True
    app.attach_session(session)
    app.console.print(
        f"Resumed [cyan]{session.meta.id}[/] — {session.meta.title} "
        f"({len(app.agent.messages)} messages restored)"
    )
    return True


@command("instructions", "Show the active project instruction file (LOCALCODE.md / CLAUDE.md)")
async def _instructions(app: App, args: str) -> bool:
    from localcode.agent.prompts import load_project_instructions
    from rich.markdown import Markdown

    result = load_project_instructions(app.root)
    if result is None:
        app.console.print(
            "[dim]No project instruction file found. "
            "Create LOCALCODE.md in the project root to add repository-specific instructions.[/]"
        )
    else:
        name, text = result
        app.console.print(f"[bold]Project instructions[/] ([cyan]{name}[/]):\n")
        app.console.print(Markdown(text))
    return True


@command("undo", "Revert the file changes made by the last turn")
async def _undo(app: App, args: str) -> bool:
    if app.checkpoints is None:
        app.console.print("[dim]Checkpointing is not active.[/]")
        return True
    restored = app.checkpoints.undo_last()
    if not restored:
        app.console.print("[dim]Nothing to undo.[/]")
    else:
        app.console.print("Restored:\n" + "\n".join(f"  {r}" for r in restored))
    return True


@command("history", "Show turns that modified files (undoable checkpoints)")
async def _history(app: App, args: str) -> bool:
    if app.checkpoints is None or not app.checkpoints.history():
        app.console.print("[dim]No checkpoints recorded.[/]")
        return True
    for turn_id, label, files in app.checkpoints.history():
        app.console.print(f"  #{turn_id}  {label[:70]}  [dim]({files} files)[/]")
    return True


@command("diff", "Show git diff of the working tree")
async def _diff(app: App, args: str) -> bool:
    from localcode.cli.render import print_diff
    from localcode.tools.shell import run_subprocess

    result = await run_subprocess(f"git diff {args}".strip(), str(app.root), 60, 200_000)
    print_diff(app.console, result.output)
    return True


@command("map", "Show the repository map")
async def _map(app: App, args: str) -> bool:
    if app.index is None:
        app.console.print("[dim]Index disabled.[/]")
        return True
    app.console.print(app.index.render_map(8000))
    return True


@command("init", "Re-index the repository and refresh the system prompt")
async def _init(app: App, args: str) -> bool:
    import asyncio

    if app.index is None:
        app.console.print("[dim]Index disabled.[/]")
        return True
    with app.console.status("Indexing repository…"):
        await asyncio.to_thread(app.index.scan)
    app.rebuild_system_prompt()
    app.console.print(f"[dim]Indexed {len(app.index.files)} files.[/]")
    return True


@command("config", "Show the active configuration")
async def _config(app: App, args: str) -> bool:
    dump = app.config.model_dump()
    dump["provider"]["api_key"] = "***" if dump["provider"]["api_key"] else ""
    import json

    app.console.print_json(json.dumps(dump, default=str))
    return True


@command("tools", "List available tools and their permission level")
async def _tools(app: App, args: str) -> bool:
    table = Table(box=None)
    table.add_column("tool", style="cyan")
    table.add_column("permission")
    table.add_column("description")
    styles = {"read": "green", "write": "yellow", "execute": "red"}
    for tool in sorted(app.agent.registry, key=lambda t: (t.permission, t.name)):
        first_sentence = tool.description.split(". ")[0].rstrip(".") + "."
        table.add_row(
            tool.name,
            f"[{styles[tool.permission]}]{tool.permission}[/]",
            first_sentence[:80],
        )
    app.console.print(table)
    return True


@command("commit", "Stage and commit changes with an AI-generated message: /commit [file ...]")
async def _commit(app: App, args: str) -> bool:
    from localcode.providers.base import Message
    from localcode.tools.shell import run_subprocess

    stat = await run_subprocess("git diff --staged --stat", str(app.root), 30, 10_000)
    if not stat.output.strip() or stat.output.strip() in ("(no output)", ""):
        app.console.print("[dim]Nothing staged. Use !git add <files> to stage changes first.[/]")
        return True
    app.console.print(stat.output)
    diff = await run_subprocess("git diff --staged", str(app.root), 30, 50_000)
    with app.console.status("Generating commit message…"):
        msg = await app.agent.provider.complete([
            Message(
                role="system",
                content="Write a conventional commit message. Reply with ONLY the message, nothing else.",
            ),
            Message(
                role="user",
                content=f"Write a commit message for this diff:\n\n{diff.output[:8000]}",
            ),
        ])
    msg = msg.strip().strip('"').strip("'")
    app.console.print(f"\nProposed: [bold green]{msg}[/]\n")
    answer = (
        await asyncio.to_thread(
            app.console.input,
            r"[bold]Commit?[/] [green]\[y]es[/] / [dim]Enter=accept[/] / type custom / [red]\[n]o[/]: ",
        )
    ).strip()
    if answer.lower() in ("n", "no"):
        app.console.print("[dim]Aborted.[/]")
        return True
    if answer and answer.lower() not in ("y", "yes"):
        msg = answer
    result = await run_subprocess(
        f"git commit -m {shlex.quote(msg)}", str(app.root), 30, 10_000
    )
    app.console.print(result.output)
    return True


@command("pr", "Generate a pull request description from current branch changes: /pr [base]")
async def _pr(app: App, args: str) -> bool:
    from localcode.providers.base import Message
    from localcode.tools.shell import run_subprocess

    base = args.strip() or "main"
    branch = await run_subprocess("git branch --show-current", str(app.root), 10, 500)
    log_out = await run_subprocess(
        f"git log --oneline {base}..HEAD", str(app.root), 10, 5_000
    )
    if (not log_out.output.strip() or log_out.output.strip() == "(no output)") and base == "main":
        # try master as fallback
        log_out = await run_subprocess(
            "git log --oneline master..HEAD", str(app.root), 10, 5_000
        )
        base = "master"
    if not log_out.output.strip() or log_out.output.strip() == "(no output)":
        app.console.print(f"[dim]No commits ahead of '{base}'. Nothing to describe.[/]")
        return True
    stat = await run_subprocess(
        f"git diff {base}...HEAD --stat", str(app.root), 10, 5_000
    )
    with app.console.status("Generating PR description…"):
        desc = await app.agent.provider.complete([
            Message(
                role="system",
                content=(
                    "Write a GitHub pull request description in markdown. "
                    "Include a ## Summary section (3–5 bullet points) and a ## Test plan "
                    "section. Reply with ONLY the markdown."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Branch: {branch.output.strip()}\n\n"
                    f"Commits:\n{log_out.output}\n\nFiles changed:\n{stat.output}"
                ),
            ),
        ])
    app.console.print(desc.strip())
    app.console.print()
    # copy to clipboard if possible
    try:
        import subprocess

        for cmd in (
            ["xclip", "-i", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
            ["wl-copy"],
        ):
            r = subprocess.run(cmd, input=desc.encode(), capture_output=True, timeout=3)
            if r.returncode == 0:
                app.console.print("[dim]Copied to clipboard.[/]")
                break
    except Exception:  # noqa: BLE001
        pass
    return True


@command("doctor", "Health check: provider, model, tools, config, index")
async def _doctor(app: App, args: str) -> bool:
    from localcode.providers.base import Message

    table = Table(box=None)
    table.add_column("check", style="cyan")
    table.add_column("status")
    table.add_column("detail")

    with app.console.status("Checking provider…"):
        try:
            models = await app.agent.provider.list_models()
            table.add_row("provider", "[green]OK[/]", f"{len(models)} model(s) available")
        except Exception as e:  # noqa: BLE001
            table.add_row("provider", "[red]FAIL[/]", str(e)[:80])

    with app.console.status("Pinging model…"):
        try:
            reply = await app.agent.provider.complete(
                [Message(role="user", content="Reply with one word: ready")]
            )
            table.add_row("model", "[green]OK[/]", reply.strip()[:60])
        except Exception as e:  # noqa: BLE001
            table.add_row("model", "[red]FAIL[/]", str(e)[:80])

    n_tools = len(list(app.agent.registry))
    table.add_row("tools", "[green]OK[/]", f"{n_tools} tools registered")
    table.add_row(
        "config",
        "[green]OK[/]",
        f"{app.config.provider.type} @ {app.config.provider.base_url}",
    )
    if app.index is None:
        table.add_row("index", "[dim]disabled[/]", "")
    else:
        table.add_row("index", "[green]OK[/]", f"{len(app.index.files)} files indexed")
    table.add_row("session", "[green]OK[/]", app.session.meta.id)
    app.console.print(table)
    return True


@command("run", "Run a prompt in the background (detached): /run <prompt>")
async def _run(app: App, args: str) -> bool:
    if not args:
        app.console.print("[red]Usage: /run <prompt>[/]")
        return True

    import time

    from localcode.agent.loop import Agent
    from localcode.config.loader import data_dir
    from localcode.session.store import SessionStore
    from localcode.tools.base import ToolContext, build_default_registry

    log_dir = data_dir() / "background"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run-{int(time.time())}.log"

    bg_session = SessionStore(data_dir(), app.root).create()
    bg_ctx = ToolContext(root=app.root, config=app.config)
    bg_registry = build_default_registry(bg_ctx)
    bg_agent = Agent(
        provider=app.agent.provider,
        registry=bg_registry,
        tool_ctx=bg_ctx,
        config=app.config,
        session=bg_session,
    )
    bg_agent.set_system_prompt(app.agent._system_prompt)
    bg_agent.mode = "full-auto"

    async def _task() -> None:
        parts: list[str] = []
        async for event in bg_agent.run_turn(args):
            if event.kind == "text" and event.text:
                parts.append(event.text)
        output = "".join(parts)
        log_path.write_text(output, encoding="utf-8")
        preview = output[:200] + ("…" if len(output) > 200 else "")
        app.console.print(
            f"\n[bold green]Background task done.[/] Log: {log_path}\n[dim]{preview}[/]"
        )

    asyncio.create_task(_task())
    app.console.print(f"[dim]Background task started. Log will be written to: {log_path}[/]")
    return True


@command("search", "Search sessions across all projects: /search <query>")
async def _search(app: App, args: str) -> bool:
    if not args:
        app.console.print("[red]Usage: /search <query>[/]")
        return True

    matches = app.store.search(args, limit=20)
    if not matches:
        app.console.print(f"[dim]No sessions matching '{args}'.[/]")
        return True

    table = Table(box=None)
    table.add_column("id", style="cyan")
    table.add_column("title")
    table.add_column("project", style="dim")
    for sid, title, project in matches:
        table.add_row(sid, title[:60], project)
    app.console.print(table)
    app.console.print("[dim]Resume with: /resume <id>[/]")
    return True


@command("worktree", "Manage git worktree isolation: /worktree [status|merge|discard]")
async def _worktree(app: App, args: str) -> bool:
    wt = getattr(app, "worktree", None)
    if wt is None or not wt.active:
        app.console.print(
            "[dim]No active worktree. Launch with --worktree to isolate agent changes.[/]"
        )
        return True
    sub = args.strip().lower() or "status"
    if sub == "status":
        app.console.print(await wt.status())
    elif sub == "merge":
        with app.console.status("Merging worktree changes back…"):
            msg = await wt.merge_back()
        app.console.print(msg)
    elif sub == "discard":
        answer = (
            await asyncio.to_thread(
                app.console.input,
                "[bold red]Discard all worktree changes?[/] [y/N]: ",
            )
        ).strip().lower()
        if answer in ("y", "yes"):
            await wt.discard()
            app.console.print("[dim]Worktree discarded.[/]")
        else:
            app.console.print("[dim]Aborted.[/]")
    else:
        app.console.print("[red]Usage: /worktree [status|merge|discard][/]")
    return True


@command("quit", "Exit localcode")
async def _quit(app: App, args: str) -> bool:
    return False


@command("exit", "Exit localcode")
async def _exit(app: App, args: str) -> bool:
    return False
