# Plugin documentation

Plugins add tools to the agent. A plugin is any module exposing:

```python
def register(registry, ctx) -> None: ...
```

where `registry` is a `ToolRegistry` and `ctx` is the shared `ToolContext`
(workspace root, config, checkpoint manager, repo index).

## Three ways to ship a plugin

### 1. Drop-in file (simplest)

Put a `.py` file in `~/.config/localcode/plugins/`:

```python
# ~/.config/localcode/plugins/wordcount.py
from localcode.tools.base import Tool, ToolResult

class WordCountTool(Tool):
    name = "word_count"
    description = "Count words in a file."
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def run(self, path: str, **_) -> ToolResult:
        text = self.ctx.resolve(path).read_text(encoding="utf-8")
        return ToolResult(ok=True, output=f"{len(text.split())} words")

def register(registry, ctx):
    registry.register(WordCountTool(ctx))
```

It loads on next startup; verify with `/config` or ask the model to use it.

### 2. Config-listed module

```toml
# .localcode.toml
plugins = ["mycompany.localcode_tools"]
```

The module must be importable (installed in the same environment) and
expose `register`.

### 3. Installable package (entry points)

```toml
# your package's pyproject.toml
[project.entry-points."localcode.tools"]
mytools = "mypackage.plugin:register"
```

`pip install` the package and localcode discovers it automatically.

## Writing good tools

- **Permissions matter.** `read` runs without prompting; `write` triggers
  the approval gate and should call `self.ctx.before_mutation(path)` before
  touching each file so `/undo` works; `execute` is for anything that runs
  code or affects state outside the workspace.
- **Describe for the model.** `description` and the JSON-schema field
  descriptions are what the model reads to decide when and how to call
  your tool. Be explicit about argument formats.
- **Return errors, don't raise.** `ToolResult.error("helpful message")`
  lets the model recover. Exceptions are caught and reported, but read as
  crashes.
- **Bound your output.** Use `localcode.tools.base.truncate(text,
  self.ctx.max_output_chars())` for potentially large results.
- **Async-friendly.** `run()` is `async`; use `asyncio.create_subprocess_*`
  (see `tools/shell.py:run_subprocess`) rather than blocking calls.

## Failure isolation

A plugin that raises during import or `register()` logs a warning and is
skipped — it cannot break startup. Check
`~/.local/share/localcode/logs/localcode.log` when a plugin doesn't appear.
