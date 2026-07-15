from pathlib import Path

import pytest

from localcode.config.schema import IndexConfig
from localcode.index.detect import detect_project
from localcode.index.graph import pagerank, resolve_import
from localcode.index.imports import extract
from localcode.index.repomap import RepoIndex


def test_extract_python() -> None:
    imports, symbols = extract(
        "import os\nfrom mypkg.mod import thing\n\nclass Foo:\n    pass\n\ndef bar():\n    pass\n",
        "python",
        use_tree_sitter=False,
    )
    assert "os" in imports and "mypkg.mod" in imports
    assert "Foo" in symbols and "bar" in symbols


def test_extract_javascript() -> None:
    imports, symbols = extract(
        "import x from './x';\nconst y = require('./y');\nexport function go() {}\nexport class App {}\n",
        "javascript",
        use_tree_sitter=False,
    )
    assert "./x" in imports and "./y" in imports
    assert "go" in symbols and "App" in symbols


def test_resolve_python_import() -> None:
    files = {"pkg/mod.py", "pkg/__init__.py", "main.py"}
    assert resolve_import("main.py", "pkg.mod", files, "python") == "pkg/mod.py"
    assert resolve_import("main.py", "pkg", files, "python") == "pkg/__init__.py"
    assert resolve_import("main.py", "missing", files, "python") is None


def test_resolve_js_relative() -> None:
    files = {"src/a.ts", "src/lib/b.ts", "src/lib/index.ts"}
    assert resolve_import("src/a.ts", "./lib/b", files, "typescript") == "src/lib/b.ts"
    assert resolve_import("src/a.ts", "./lib", files, "typescript") == "src/lib/index.ts"
    assert resolve_import("src/a.ts", "react", files, "typescript") is None


def test_pagerank_favors_depended_upon() -> None:
    nodes = {"a", "b", "core"}
    edges = {"a": {"core"}, "b": {"core"}}
    ranks = pagerank(edges, nodes)
    assert ranks["core"] > ranks["a"]


def test_detect_python_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "main.py").write_text("x = 1\n")
    info = detect_project(tmp_path)
    assert "pyproject" in info.build_systems
    assert info.test_command == "python -m pytest"


def test_detect_node_project(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"scripts": {"test": "jest", "build": "tsc"}}'
    )
    (tmp_path / "index.js").write_text("")
    info = detect_project(tmp_path)
    assert "jest" in info.test_frameworks
    assert info.build_command == "npm run build"


def test_repo_index_scan_and_map(workspace: Path, tmp_path: Path) -> None:
    index = RepoIndex(workspace, IndexConfig(), tmp_path / "cache")
    index.scan()
    assert "src/app.py" in index.files
    assert "src/util.py" in index.files
    # util is imported by app -> should rank at least as high
    assert index.files["src/util.py"].rank >= index.files["src/app.py"].rank
    body = index.render_map(4000)
    assert "src/util.py" in body and "python" in body


def test_repo_index_incremental_cache(workspace: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    # cache must live outside the workspace or it would itself get indexed
    cache = tmp_path_factory.mktemp("cache")
    index1 = RepoIndex(workspace, IndexConfig(), cache)
    index1.scan()
    # second index re-uses the mtime cache
    index2 = RepoIndex(workspace, IndexConfig(), cache)
    index2.scan()
    assert index2.files.keys() == index1.files.keys()
    # touching a file re-parses it
    (workspace / "src/util.py").write_text("def helper():\n    return 1\n\ndef extra():\n    pass\n")
    index2.scan()
    assert "extra" in index2.files["src/util.py"].symbols


def test_relevant_files(workspace: Path, tmp_path: Path) -> None:
    index = RepoIndex(workspace, IndexConfig(), None)
    index.scan()
    ranked = index.relevant_files("helper util")
    assert ranked
    assert ranked[0][0] == "src/util.py"


def test_gitignore_respected(workspace: Path) -> None:
    (workspace / ".gitignore").write_text("secret/\n")
    (workspace / "secret").mkdir()
    (workspace / "secret" / "hidden.py").write_text("x = 1")
    index = RepoIndex(workspace, IndexConfig(), None)
    index.scan()
    assert "secret/hidden.py" not in index.files
