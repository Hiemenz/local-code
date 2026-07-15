"""Repository-intelligence tools backed by the index."""

from __future__ import annotations

from typing import Any

from localcode.tools.base import Tool, ToolResult, truncate


class RepoMapTool(Tool):
    name = "repo_map"
    description = (
        "Show a ranked map of the repository: project type, build/test commands, "
        "and the most important files with their top-level symbols."
    )
    permission = "read"
    parameters = {"type": "object", "properties": {}}

    async def run(self, **_: Any) -> ToolResult:
        if self.ctx.index is None:
            return ToolResult.error("Repository index is disabled.")
        body = self.ctx.index.render_map(self.ctx.config.context.repo_map_tokens * 4)
        return ToolResult(ok=True, output=body)


class FindRelevantFilesTool(Tool):
    name = "find_relevant_files"
    description = (
        "Rank repository files by relevance to a query (matches names, symbols, "
        "and dependency importance). Use before diving into unfamiliar code."
    )
    permission = "read"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 15},
        },
        "required": ["query"],
    }

    async def run(self, query: str, limit: int = 15, **_: Any) -> ToolResult:
        if self.ctx.index is None:
            return ToolResult.error("Repository index is disabled.")
        ranked = self.ctx.index.relevant_files(query, limit=limit)
        if not ranked:
            return ToolResult(ok=True, output="No relevant files found.")
        lines = [f"{score:6.2f}  {path}" for path, score in ranked]
        return ToolResult(
            ok=True,
            output=truncate("\n".join(lines), self.ctx.max_output_chars()),
            data={"files": [p for p, _ in ranked]},
        )


TOOLS = [RepoMapTool, FindRelevantFilesTool]
