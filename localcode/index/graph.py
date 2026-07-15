"""Dependency graph: resolve imports to repository files and rank with PageRank."""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

_JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


def resolve_import(importer: str, spec: str, files: set[str], lang: str) -> str | None:
    """Map an import spec to a repository-relative file path, or None."""
    if lang == "python":
        parts = spec.split(".")
        for depth in range(len(parts), 0, -1):
            base = "/".join(parts[:depth])
            for candidate in (f"{base}.py", f"{base}/__init__.py"):
                if candidate in files:
                    return candidate
            # also try relative to the importer's package
            pkg = str(PurePosixPath(importer).parent)
            if pkg != ".":
                for candidate in (f"{pkg}/{base}.py", f"{pkg}/{base}/__init__.py"):
                    if candidate in files:
                        return candidate
        return None
    if lang in ("javascript", "typescript"):
        if not spec.startswith("."):
            return None  # bare specifier = external package
        target = PurePosixPath(importer).parent / spec
        # normalize ".." segments
        norm_parts: list[str] = []
        for part in target.parts:
            if part == "..":
                if norm_parts:
                    norm_parts.pop()
            elif part != ".":
                norm_parts.append(part)
        norm = "/".join(norm_parts)
        candidates = [norm] + [norm + ext for ext in _JS_EXTS]
        candidates += [f"{norm}/index{ext}" for ext in _JS_EXTS]
        for candidate in candidates:
            if candidate in files:
                return candidate
        return None
    if lang in ("c", "cpp"):
        parent = PurePosixPath(importer).parent
        for candidate in (spec, str(parent / spec)):
            if candidate in files:
                return candidate
        return None
    if lang == "rust":
        head = spec.split("::")[0]
        if head in ("crate", "super", "self", "std", "core", "alloc"):
            head = spec.split("::")[1] if "::" in spec else ""
        for candidate in (f"src/{head}.rs", f"src/{head}/mod.rs", f"{head}.rs"):
            if candidate in files:
                return candidate
        return None
    return None


def build_edges(
    imports_by_file: dict[str, list[str]], langs: dict[str, str]
) -> dict[str, set[str]]:
    """file -> set of files it imports."""
    files = set(imports_by_file)
    edges: dict[str, set[str]] = defaultdict(set)
    for path, specs in imports_by_file.items():
        lang = langs.get(path, "")
        for spec in specs:
            resolved = resolve_import(path, spec, files, lang)
            if resolved and resolved != path:
                edges[path].add(resolved)
    return dict(edges)


def pagerank(
    edges: dict[str, set[str]], nodes: set[str], damping: float = 0.85, iterations: int = 20
) -> dict[str, float]:
    """Rank files by how often they are depended upon."""
    if not nodes:
        return {}
    rank = dict.fromkeys(nodes, 1.0 / len(nodes))
    incoming: dict[str, list[str]] = defaultdict(list)
    out_degree: dict[str, int] = {}
    for src, targets in edges.items():
        valid = [t for t in targets if t in nodes]
        out_degree[src] = len(valid)
        for t in valid:
            incoming[t].append(src)
    base = (1.0 - damping) / len(nodes)
    for _ in range(iterations):
        new_rank: dict[str, float] = {}
        for node in nodes:
            score = sum(
                rank[src] / out_degree[src] for src in incoming.get(node, []) if out_degree.get(src)
            )
            new_rank[node] = base + damping * score
        rank = new_rank
    return rank
