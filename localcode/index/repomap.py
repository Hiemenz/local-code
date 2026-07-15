"""RepoIndex: incremental, cached repository index and ranked repo map."""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pathspec

from localcode.config.schema import IndexConfig
from localcode.index import graph
from localcode.index.detect import EXT_LANG, ProjectInfo, detect_project
from localcode.index.imports import extract
from localcode.logging import get_logger

log = get_logger("index")

_ALWAYS_SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox",
                ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", "target"}

CACHE_VERSION = 2


@dataclass
class FileInfo:
    path: str  # relative, posix-style
    mtime: float
    size: int
    lang: str
    imports: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    rank: float = 0.0


class RepoIndex:
    def __init__(self, root: Path, config: IndexConfig, cache_dir: Path | None = None) -> None:
        self.root = root.resolve()
        self.config = config
        self.files: dict[str, FileInfo] = {}
        self.project: ProjectInfo = ProjectInfo()
        self._cache_file: Path | None = None
        if cache_dir is not None:
            digest = hashlib.sha1(str(self.root).encode()).hexdigest()[:16]
            self._cache_file = cache_dir / "index" / f"{digest}.json"
        self._ignore_spec = self._build_ignore_spec()

    # -- scanning ---------------------------------------------------------

    def _build_ignore_spec(self) -> pathspec.PathSpec:  # type: ignore[type-arg]
        patterns = list(self.config.ignore)
        gitignore = self.root / ".gitignore"
        if gitignore.exists():
            with contextlib.suppress(OSError):
                patterns += gitignore.read_text(encoding="utf-8").splitlines()
        return pathspec.PathSpec.from_lines("gitignore", [p for p in patterns if p.strip()])

    def _iter_source_files(self) -> list[Path]:
        out: list[Path] = []
        stack = [self.root]
        while stack:
            directory = stack.pop()
            try:
                entries = sorted(directory.iterdir())
            except OSError:
                continue
            for entry in entries:
                rel = entry.relative_to(self.root).as_posix()
                if entry.is_dir():
                    if entry.name in _ALWAYS_SKIP or self._ignore_spec.match_file(rel + "/"):
                        continue
                    stack.append(entry)
                elif entry.is_file():
                    if self._ignore_spec.match_file(rel):
                        continue
                    if entry.suffix.lower() not in EXT_LANG:
                        continue
                    try:
                        if entry.stat().st_size > self.config.max_file_size:
                            continue
                    except OSError:
                        continue
                    out.append(entry)
        return out

    def scan(self) -> None:
        """Full or incremental scan; only re-parses files whose mtime/size changed."""
        started = time.monotonic()
        cached = self._load_cache()
        new_files: dict[str, FileInfo] = {}
        parsed = 0
        for path in self._iter_source_files():
            rel = path.relative_to(self.root).as_posix()
            try:
                stat = path.stat()
            except OSError:
                continue
            prior = cached.get(rel)
            if prior is not None and prior.mtime == stat.st_mtime and prior.size == stat.st_size:
                new_files[rel] = prior
                continue
            lang = EXT_LANG.get(path.suffix.lower(), "")
            info = FileInfo(path=rel, mtime=stat.st_mtime, size=stat.st_size, lang=lang)
            if lang not in ("markdown", "json", "yaml", "toml", "html", "css", "sql"):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    info.imports, info.symbols = extract(
                        text, lang, use_tree_sitter=self.config.use_tree_sitter
                    )
                    parsed += 1
                except OSError:
                    pass
            new_files[rel] = info
        self.files = new_files
        self._compute_ranks()
        counts: dict[str, int] = {}
        for info in self.files.values():
            counts[info.lang] = counts.get(info.lang, 0) + 1
        self.project = detect_project(self.root, counts)
        self._save_cache()
        log.info(
            "indexed repository",
            extra={"data": {"files": len(self.files), "parsed": parsed,
                            "seconds": round(time.monotonic() - started, 2)}},
        )

    def refresh(self, paths: list[Path] | None = None) -> None:
        """Cheap re-scan; the mtime cache makes this fast after edits."""
        self.scan()

    def _compute_ranks(self) -> None:
        imports_by_file = {p: fi.imports for p, fi in self.files.items()}
        langs = {p: fi.lang for p, fi in self.files.items()}
        edges = graph.build_edges(imports_by_file, langs)
        ranks = graph.pagerank(edges, set(self.files))
        for path, info in self.files.items():
            info.rank = ranks.get(path, 0.0)

    # -- cache --------------------------------------------------------------

    def _load_cache(self) -> dict[str, FileInfo]:
        if self._cache_file is None or not self._cache_file.exists():
            return {}
        try:
            data = json.loads(self._cache_file.read_text(encoding="utf-8"))
            if data.get("version") != CACHE_VERSION or data.get("root") != str(self.root):
                return {}
            return {
                entry["path"]: FileInfo(**entry) for entry in data.get("files", [])
            }
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            return {}

    def _save_cache(self) -> None:
        if self._cache_file is None:
            return
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": CACHE_VERSION,
                "root": str(self.root),
                "files": [asdict(fi) for fi in self.files.values()],
            }
            self._cache_file.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as e:
            log.warning(f"could not write index cache: {e}")

    # -- queries -------------------------------------------------------------

    def relevant_files(self, query: str, limit: int = 15) -> list[tuple[str, float]]:
        """Rank files by query-term matches on path/symbols plus graph importance."""
        terms = [t.lower() for t in query.replace("_", " ").replace("/", " ").split() if len(t) > 1]
        max_rank = max((fi.rank for fi in self.files.values()), default=1.0) or 1.0
        scored: list[tuple[str, float]] = []
        for path, info in self.files.items():
            score = (info.rank / max_rank) * 2.0
            path_lower = path.lower()
            symbols_lower = " ".join(info.symbols).lower()
            for term in terms:
                if term in path_lower:
                    score += 3.0
                if term in symbols_lower:
                    score += 2.0
            if terms and score <= (info.rank / max_rank) * 2.0:
                continue  # no term matched; only include when query is empty
            scored.append((path, round(score, 2)))
        if not terms:
            scored = [(p, round(fi.rank / max_rank * 10, 2)) for p, fi in self.files.items()]
        scored.sort(key=lambda kv: -kv[1])
        return scored[:limit]

    def render_map(self, char_budget: int = 8000) -> str:
        """Human/model-readable repository map, most important files first."""
        lines: list[str] = [self.project.summary(), "", f"Files indexed: {len(self.files)}", ""]
        ranked = sorted(self.files.values(), key=lambda fi: -fi.rank)
        used = sum(len(line) + 1 for line in lines)
        for info in ranked:
            symbols = ", ".join(info.symbols[:8])
            entry = f"{info.path}" + (f": {symbols}" if symbols else "")
            if used + len(entry) > char_budget:
                lines.append(f"... ({len(ranked) - lines.count(chr(10))} more files)")
                break
            lines.append(entry)
            used += len(entry) + 1
        return "\n".join(lines)
