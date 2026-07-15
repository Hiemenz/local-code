"""Project detection: languages, build systems, package managers, test frameworks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

EXT_LANG: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cxx": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".sh": "shell", ".bash": "shell",
    ".lua": "lua",
    ".sql": "sql",
    ".html": "html", ".css": "css", ".scss": "css",
    ".md": "markdown", ".rst": "markdown",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".json": "json",
}


@dataclass
class ProjectInfo:
    languages: dict[str, int] = field(default_factory=dict)  # language -> file count
    build_systems: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    test_frameworks: list[str] = field(default_factory=list)
    build_command: str | None = None
    test_command: str | None = None

    def summary(self) -> str:
        langs = ", ".join(
            f"{name} ({count})"
            for name, count in sorted(self.languages.items(), key=lambda kv: -kv[1])[:5]
        )
        parts = [f"Languages: {langs or 'unknown'}"]
        if self.build_systems:
            parts.append(f"Build: {', '.join(self.build_systems)}")
        if self.package_managers:
            parts.append(f"Package manager: {', '.join(self.package_managers)}")
        if self.test_frameworks:
            parts.append(f"Tests: {', '.join(self.test_frameworks)}")
        if self.test_command:
            parts.append(f"Test command: {self.test_command}")
        if self.build_command:
            parts.append(f"Build command: {self.build_command}")
        return "\n".join(parts)


# marker file -> (build system, package manager)
_MARKERS: list[tuple[str, str | None, str | None]] = [
    ("pyproject.toml", "pyproject", None),
    ("setup.py", "setuptools", None),
    ("uv.lock", None, "uv"),
    ("poetry.lock", None, "poetry"),
    ("Pipfile", None, "pipenv"),
    ("requirements.txt", None, "pip"),
    ("package.json", "npm scripts", None),
    ("pnpm-lock.yaml", None, "pnpm"),
    ("yarn.lock", None, "yarn"),
    ("package-lock.json", None, "npm"),
    ("bun.lockb", None, "bun"),
    ("Cargo.toml", "cargo", "cargo"),
    ("go.mod", "go", "go modules"),
    ("Makefile", "make", None),
    ("CMakeLists.txt", "cmake", None),
    ("pom.xml", "maven", "maven"),
    ("build.gradle", "gradle", "gradle"),
    ("build.gradle.kts", "gradle", "gradle"),
    ("Gemfile", "bundler", "bundler"),
    ("mix.exs", "mix", "hex"),
    ("composer.json", "composer", "composer"),
]


def _package_json_scripts(root: Path) -> dict[str, str]:
    pkg = root / "package.json"
    if not pkg.exists():
        return {}
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    scripts = data.get("scripts")
    return scripts if isinstance(scripts, dict) else {}


def detect_project(root: Path, language_counts: dict[str, int] | None = None) -> ProjectInfo:
    info = ProjectInfo()
    if language_counts is not None:
        info.languages = dict(language_counts)
    else:
        for p in root.rglob("*"):
            if p.is_file() and not {".git", "node_modules", ".venv", "__pycache__"} & set(p.parts):
                lang = EXT_LANG.get(p.suffix.lower())
                if lang:
                    info.languages[lang] = info.languages.get(lang, 0) + 1

    for marker, build, pkg_mgr in _MARKERS:
        if (root / marker).exists():
            if build and build not in info.build_systems:
                info.build_systems.append(build)
            if pkg_mgr and pkg_mgr not in info.package_managers:
                info.package_managers.append(pkg_mgr)

    _detect_python(root, info)
    _detect_node(root, info)
    _detect_others(root, info)
    return info


def _detect_python(root: Path, info: ProjectInfo) -> None:
    if "python" not in info.languages:
        return
    has_pytest = any(
        (root / f).exists() for f in ("pytest.ini", "conftest.py", "tests/conftest.py")
    )
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            text = pyproject.read_text(encoding="utf-8")
            has_pytest = has_pytest or "pytest" in text
        except OSError:
            pass
    if has_pytest or (root / "tests").is_dir():
        info.test_frameworks.append("pytest")
        info.test_command = info.test_command or "python -m pytest"
    elif (root / "test").is_dir():
        info.test_frameworks.append("unittest")
        info.test_command = info.test_command or "python -m unittest discover"


def _detect_node(root: Path, info: ProjectInfo) -> None:
    scripts = _package_json_scripts(root)
    if not scripts:
        return
    if "test" in scripts:
        runner = scripts["test"]
        for fw in ("jest", "vitest", "mocha", "ava", "tap"):
            if fw in runner:
                info.test_frameworks.append(fw)
                break
        else:
            info.test_frameworks.append("npm test")
        info.test_command = info.test_command or "npm test --silent"
    if "build" in scripts:
        info.build_command = info.build_command or "npm run build"


def _detect_others(root: Path, info: ProjectInfo) -> None:
    if (root / "Cargo.toml").exists():
        info.test_frameworks.append("cargo test")
        info.test_command = info.test_command or "cargo test"
        info.build_command = info.build_command or "cargo build"
    if (root / "go.mod").exists():
        info.test_frameworks.append("go test")
        info.test_command = info.test_command or "go test ./..."
        info.build_command = info.build_command or "go build ./..."
    if (root / "Makefile").exists() and not info.build_command:
        info.build_command = "make"
    if (root / "CMakeLists.txt").exists() and not info.build_command:
        info.build_command = "cmake --build build"
    if (root / "pom.xml").exists():
        info.test_command = info.test_command or "mvn test"
        info.build_command = info.build_command or "mvn package"
    if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        info.test_command = info.test_command or "./gradlew test"
        info.build_command = info.build_command or "./gradlew build"
