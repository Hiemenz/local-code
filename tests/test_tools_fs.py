import pytest

from localcode.errors import PathSecurityError
from localcode.tools.base import ToolContext
from localcode.tools.fs import (
    ApplyPatchTool,
    EditFileTool,
    GlobTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)


async def test_read_file(tool_ctx: ToolContext) -> None:
    result = await ReadFileTool(tool_ctx).run(path="src/app.py")
    assert result.ok
    assert "import util" in result.output
    assert result.output.lstrip().startswith("1")  # line numbers


async def test_read_missing(tool_ctx: ToolContext) -> None:
    result = await ReadFileTool(tool_ctx).run(path="nope.py")
    assert not result.ok


async def test_read_offset_limit(tool_ctx: ToolContext) -> None:
    result = await ReadFileTool(tool_ctx).run(path="src/app.py", offset=4, limit=1)
    assert result.ok
    assert "def main" in result.output
    assert "import util" not in result.output


async def test_write_creates_dirs_and_snapshots(tool_ctx: ToolContext) -> None:
    result = await WriteFileTool(tool_ctx).run(path="new/dir/file.txt", content="hi")
    assert result.ok
    assert (tool_ctx.root / "new/dir/file.txt").read_text() == "hi"
    assert tool_ctx.files_modified


async def test_path_escape_blocked(tool_ctx: ToolContext) -> None:
    with pytest.raises(PathSecurityError):
        tool_ctx.resolve("../../etc/passwd")


async def test_edit_file_unique(tool_ctx: ToolContext) -> None:
    result = await EditFileTool(tool_ctx).run(
        path="src/util.py", old_string="return 42", new_string="return 43"
    )
    assert result.ok
    assert "return 43" in (tool_ctx.root / "src/util.py").read_text()


async def test_edit_file_ambiguous_rejected(tool_ctx: ToolContext) -> None:
    (tool_ctx.root / "dup.py").write_text("x = 1\nx = 1\n")
    result = await EditFileTool(tool_ctx).run(path="dup.py", old_string="x = 1", new_string="y = 2")
    assert not result.ok
    assert "2 times" in result.output


async def test_edit_file_replace_all(tool_ctx: ToolContext) -> None:
    (tool_ctx.root / "dup.py").write_text("x = 1\nx = 1\n")
    result = await EditFileTool(tool_ctx).run(
        path="dup.py", old_string="x = 1", new_string="y = 2", replace_all=True
    )
    assert result.ok
    assert (tool_ctx.root / "dup.py").read_text() == "y = 2\ny = 2\n"


async def test_edit_not_found_message(tool_ctx: ToolContext) -> None:
    result = await EditFileTool(tool_ctx).run(
        path="src/util.py", old_string="does not exist", new_string="x"
    )
    assert not result.ok and "not found" in result.output


async def test_apply_patch_modify(tool_ctx: ToolContext) -> None:
    patch = """--- a/src/util.py
+++ b/src/util.py
@@ -1,2 +1,2 @@
 def helper():
-    return 42
+    return 99
"""
    result = await ApplyPatchTool(tool_ctx).run(patch=patch)
    assert result.ok, result.output
    assert "return 99" in (tool_ctx.root / "src/util.py").read_text()


async def test_apply_patch_create_and_delete(tool_ctx: ToolContext) -> None:
    create = """--- /dev/null
+++ b/newmod.py
@@ -0,0 +1,2 @@
+def created():
+    return True
"""
    result = await ApplyPatchTool(tool_ctx).run(patch=create)
    assert result.ok, result.output
    assert (tool_ctx.root / "newmod.py").read_text() == "def created():\n    return True\n"

    delete = """--- a/newmod.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def created():
-    return True
"""
    result = await ApplyPatchTool(tool_ctx).run(patch=delete)
    assert result.ok, result.output
    assert not (tool_ctx.root / "newmod.py").exists()


async def test_apply_patch_bad_context(tool_ctx: ToolContext) -> None:
    patch = """--- a/src/util.py
+++ b/src/util.py
@@ -1,2 +1,2 @@
 def helper():
-    return 1000000
+    return 99
"""
    result = await ApplyPatchTool(tool_ctx).run(patch=patch)
    assert not result.ok


async def test_list_dir(tool_ctx: ToolContext) -> None:
    result = await ListDirTool(tool_ctx).run(path="src")
    assert result.ok
    assert "app.py" in result.output


async def test_glob(tool_ctx: ToolContext) -> None:
    result = await GlobTool(tool_ctx).run(pattern="src/**/*.py")
    assert result.ok
    assert "src/app.py" in result.output
