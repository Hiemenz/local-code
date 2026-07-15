"""Deeper coverage of detection, import extraction, and graph resolution."""

from __future__ import annotations

from pathlib import Path

from localcode.config.schema import IndexConfig
from localcode.index.detect import detect_project
from localcode.index.graph import build_edges, resolve_import
from localcode.index.imports import extract
from localcode.index.repomap import RepoIndex

# -- import/symbol extraction per language ------------------------------------

def test_extract_rust() -> None:
    imports, symbols = extract(
        "use crate::parser::Token;\nmod lexer;\n\npub fn parse() {}\n"
        "pub struct Ast {}\npub enum Node {}\npub trait Visit {}\n",
        "rust", use_tree_sitter=False,
    )
    assert "crate::parser::Token" in imports and "lexer" in imports
    assert {"parse", "Ast", "Node", "Visit"} <= set(symbols)


def test_extract_go() -> None:
    imports, symbols = extract(
        'import "fmt"\n\nfunc Handle(w http.ResponseWriter) {}\n'
        "func (s *Server) Start() {}\ntype Server struct{}\n",
        "go", use_tree_sitter=False,
    )
    assert "fmt" in imports
    assert {"Handle", "Start", "Server"} <= set(symbols)


def test_extract_java() -> None:
    imports, symbols = extract(
        "import java.util.List;\nimport static org.junit.Assert.*;\n\n"
        "public class UserService {}\ninterface Repo {}\n",
        "java", use_tree_sitter=False,
    )
    assert "java.util.List" in imports
    assert "UserService" in symbols and "Repo" in symbols


def test_extract_ruby() -> None:
    imports, symbols = extract(
        "require 'json'\nrequire_relative 'helper'\n\n"
        "class Parser\n  def self.parse\n  end\n  def tokenize\n  end\nend\n",
        "ruby", use_tree_sitter=False,
    )
    assert "json" in imports and "helper" in imports
    assert {"Parser", "parse", "tokenize"} <= set(symbols)


def test_extract_c_includes() -> None:
    imports, _ = extract('#include "util.h"\n#include <stdio.h>\n', "c", use_tree_sitter=False)
    assert "util.h" in imports
    assert "stdio.h" not in imports  # system includes are not repo files


def test_extract_unknown_language() -> None:
    assert extract("whatever", "cobol", use_tree_sitter=False) == ([], [])


def test_extract_dedupes_and_caps() -> None:
    text = "import os\n" * 50 + "\n".join(f"def f{i}(): pass" for i in range(200))
    imports, symbols = extract(text, "python", use_tree_sitter=False)
    assert imports == ["os"]
    assert len(symbols) == 100  # capped


# -- graph resolution ----------------------------------------------------------

def test_resolve_rust_modules() -> None:
    files = {"src/lexer.rs", "src/parser/mod.rs", "main.rs"}
    assert resolve_import("src/main.rs", "lexer::Token", files, "rust") == "src/lexer.rs"
    assert resolve_import("src/main.rs", "crate::parser::x", files, "rust") == "src/parser/mod.rs"
    assert resolve_import("src/main.rs", "std::io", files, "rust") is None


def test_resolve_c_includes() -> None:
    files = {"src/util.h", "src/main.c"}
    assert resolve_import("src/main.c", "util.h", files, "c") == "src/util.h"
    assert resolve_import("src/main.c", "missing.h", files, "c") is None


def test_resolve_python_relative_to_package() -> None:
    files = {"pkg/a.py", "pkg/b.py"}
    assert resolve_import("pkg/a.py", "b", files, "python") == "pkg/b.py"


def test_build_edges_skips_self_and_unresolved() -> None:
    edges = build_edges(
        {"a.py": ["b", "external_lib", "a"], "b.py": []},
        {"a.py": "python", "b.py": "python"},
    )
    assert edges == {"a.py": {"b.py"}}


# -- project detection branches ---------------------------------------------------

def test_detect_rust(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    (tmp_path / "main.rs").write_text("fn main() {}\n")
    info = detect_project(tmp_path)
    assert "cargo" in info.build_systems
    assert info.test_command == "cargo test"
    assert info.build_command == "cargo build"


def test_detect_go(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/x\n")
    (tmp_path / "main.go").write_text("package main\n")
    info = detect_project(tmp_path)
    assert info.test_command == "go test ./..."
    assert "go modules" in info.package_managers


def test_detect_makefile_and_gradle(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text("all:\n")
    info = detect_project(tmp_path)
    assert info.build_command == "make"
    (tmp_path / "build.gradle").write_text("")
    info = detect_project(tmp_path)
    assert info.test_command == "./gradlew test"


def test_detect_maven(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project/>")
    info = detect_project(tmp_path)
    assert info.test_command == "mvn test"
    assert info.build_command == "mvn package"


def test_detect_broken_package_json(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{invalid json")
    info = detect_project(tmp_path)  # must not raise
    assert "npm scripts" in info.build_systems


def test_summary_renders_everything(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    (tmp_path / "lib.rs").write_text("pub fn x() {}\n")
    summary = detect_project(tmp_path).summary()
    assert "rust" in summary
    assert "cargo test" in summary
    assert "cargo build" in summary


# -- repo map edge cases -------------------------------------------------------------

def test_render_map_respects_budget(workspace: Path) -> None:
    index = RepoIndex(workspace, IndexConfig(), None)
    index.scan()
    small = index.render_map(char_budget=120)
    assert len(small) <= 400  # header + truncation marker only


def test_relevant_files_empty_query_uses_rank(workspace: Path) -> None:
    index = RepoIndex(workspace, IndexConfig(), None)
    index.scan()
    ranked = index.relevant_files("")
    assert len(ranked) == len(index.files)


def test_markdown_files_indexed_without_parsing(workspace: Path) -> None:
    index = RepoIndex(workspace, IndexConfig(), None)
    index.scan()
    assert "README.md" in index.files
    assert index.files["README.md"].symbols == []


def test_config_ignore_patterns(workspace: Path) -> None:
    (workspace / "generated.py").write_text("x = 1")
    cfg = IndexConfig(ignore=["generated.py"])
    index = RepoIndex(workspace, cfg, None)
    index.scan()
    assert "generated.py" not in index.files
    assert "src/app.py" in index.files
