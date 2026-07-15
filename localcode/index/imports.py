"""Import and symbol extraction. Uses tree-sitter when installed
(pip install 'localcode[treesitter]'); falls back to per-language regexes."""

from __future__ import annotations

import re
from functools import cache

from localcode.logging import get_logger

log = get_logger("index.imports")

# language -> (import regexes, symbol regexes); group 1 is the capture
_REGEXES: dict[str, tuple[list[str], list[str]]] = {
    "python": (
        [r"^\s*from\s+([\w.]+)\s+import", r"^\s*import\s+([\w.]+)"],
        [r"^(?:async\s+)?def\s+(\w+)", r"^class\s+(\w+)"],
    ),
    "javascript": (
        [r"""import\s+(?:[\w{}\s,*$]+\s+from\s+)?['"]([^'"]+)['"]""",
         r"""require\(\s*['"]([^'"]+)['"]\s*\)"""],
        [r"^(?:export\s+)?(?:async\s+)?function\s*\*?\s*(\w+)",
         r"^(?:export\s+)?class\s+(\w+)",
         r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*="],
    ),
    "go": (
        [r"""^\s*import\s+(?:\w+\s+)?"([^"]+)""", r"""^\s*(?:\w+\s+)?"([^"]+)"$"""],
        [r"^func\s+(?:\([^)]+\)\s+)?(\w+)", r"^type\s+(\w+)"],
    ),
    "rust": (
        [r"^\s*use\s+([\w:]+)", r"^\s*mod\s+(\w+)\s*;"],
        [r"^\s*(?:pub\s+)?fn\s+(\w+)", r"^\s*(?:pub\s+)?struct\s+(\w+)",
         r"^\s*(?:pub\s+)?enum\s+(\w+)", r"^\s*(?:pub\s+)?trait\s+(\w+)"],
    ),
    "java": (
        [r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;"],
        [r"(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+(\w+)"],
    ),
    "ruby": (
        [r"""^\s*require(?:_relative)?\s+['"]([^'"]+)['"]"""],
        [r"^\s*def\s+(?:self\.)?(\w+)", r"^\s*(?:class|module)\s+([\w:]+)"],
    ),
    "c": ([r'^\s*#include\s+"([^"]+)"'], []),
    "cpp": (
        [r'^\s*#include\s+"([^"]+)"'],
        [r"^\s*(?:class|struct)\s+(\w+)"],
    ),
}
_REGEXES["typescript"] = _REGEXES["javascript"]

_TS_LANG_NAMES = {
    "python": "python", "javascript": "javascript", "typescript": "typescript",
    "go": "go", "rust": "rust", "java": "java", "ruby": "ruby", "c": "c", "cpp": "cpp",
}
_TS_SYMBOL_NODES = {
    "function_definition", "class_definition",  # python
    "function_declaration", "class_declaration", "method_definition",  # js/ts
    "lexical_declaration",  # js const
    "type_declaration",  # go
    "function_item", "struct_item", "enum_item", "trait_item",  # rust
    "class_specifier", "struct_specifier",  # c/c++
    "method_declaration", "interface_declaration",  # java
}


@cache
def _get_ts_parser(lang: str):  # type: ignore[no-untyped-def]
    """Return a tree-sitter parser for lang, or None if unavailable."""
    name = _TS_LANG_NAMES.get(lang)
    if name is None:
        return None
    try:
        from tree_sitter_language_pack import get_parser

        return get_parser(name)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return None


def _extract_tree_sitter(text: str, lang: str) -> list[str] | None:
    """Top-level symbol names via tree-sitter; None if parsing unavailable."""
    parser = _get_ts_parser(lang)
    if parser is None:
        return None
    try:
        tree = parser.parse(text.encode())
        symbols: list[str] = []

        def visit(node, depth: int) -> None:  # type: ignore[no-untyped-def]
            if depth > 3:
                return
            for child in node.children:
                if child.type in _TS_SYMBOL_NODES:
                    ident = child.child_by_field_name("name")
                    if ident is None:  # e.g. lexical_declaration -> declarator
                        for sub in child.children:
                            ident = sub.child_by_field_name("name") if hasattr(sub, "child_by_field_name") else None
                            if ident is not None:
                                break
                    if ident is not None:
                        symbols.append(ident.text.decode(errors="replace"))
                visit(child, depth + 1)

        visit(tree.root_node, 0)
        return symbols
    except Exception as e:  # noqa: BLE001
        log.debug(f"tree-sitter parse failed for {lang}: {e}")
        return None


def extract(text: str, lang: str, use_tree_sitter: bool = True) -> tuple[list[str], list[str]]:
    """Return (imports, symbols) for a source file."""
    import_rx, symbol_rx = _REGEXES.get(lang, ([], []))
    imports: list[str] = []
    for pattern in import_rx:
        imports.extend(re.findall(pattern, text, re.MULTILINE))

    symbols: list[str] | None = None
    if use_tree_sitter:
        symbols = _extract_tree_sitter(text, lang)
    if symbols is None:
        symbols = []
        for pattern in symbol_rx:
            symbols.extend(re.findall(pattern, text, re.MULTILINE))

    def dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    return dedupe(imports)[:100], dedupe(symbols)[:100]
