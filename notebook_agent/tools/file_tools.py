"""Coding-agent style file reading and search tools."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from notebook_agent.security.policy import SecurityError, SecurityPolicy
from notebook_agent.tools.base import BaseTool, ToolResult


class GlobFilesTool(BaseTool):
    name = "glob_files"
    description = "Find files matching a glob pattern (e.g. '**/*.py', 'src/*.txt')."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern to match files."},
            "path": {"type": "string", "description": "Base directory to search in (default: working dir).", "default": ""},
        },
        "required": ["pattern"],
    }

    def __init__(self, working_dir: str, security: SecurityPolicy):
        self._working_dir = Path(working_dir).resolve()
        self._security = security

    async def execute(self, pattern: str, path: str = "", **kwargs: Any) -> ToolResult:
        base = self._security.resolve_path(path) if path else self._working_dir
        try:
            self._security.check_path(str(base))
        except SecurityError as e:
            return ToolResult(content=str(e), is_error=True)
        try:
            matches = sorted(base.glob(pattern))
        except Exception as e:
            return ToolResult(content=f"Glob error: {e}", is_error=True)

        # Filter by security
        results = []
        for m in matches:
            if self._security.is_path_allowed(str(m)):
                rel = m.relative_to(self._working_dir) if m.is_relative_to(self._working_dir) else m
                results.append(str(rel))

        if not results:
            return ToolResult(content=f"No files match pattern '{pattern}'.")

        if len(results) > 100:
            return ToolResult(content="\n".join(results[:100]) + f"\n... and {len(results) - 100} more files")

        return ToolResult(content="\n".join(results))


class GrepFilesTool(BaseTool):
    name = "grep_files"
    description = "Search file contents for a regex pattern. Returns matching lines with file:line:content format."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "Directory or file to search in.", "default": ""},
            "glob": {"type": "string", "description": "File glob filter (e.g. '*.py').", "default": ""},
        },
        "required": ["pattern"],
    }

    def __init__(self, working_dir: str, security: SecurityPolicy):
        self._working_dir = Path(working_dir).resolve()
        self._security = security

    async def execute(self, pattern: str, path: str = "", glob: str = "", **kwargs: Any) -> ToolResult:
        search_path = self._security.resolve_path(path) if path else self._working_dir
        try:
            self._security.check_path(str(search_path))
        except SecurityError as e:
            return ToolResult(content=str(e), is_error=True)

        # Try ripgrep first, fall back to Python
        try:
            return self._rg_search(pattern, search_path, glob)
        except FileNotFoundError:
            return self._python_search(pattern, search_path, glob)

    def _rg_search(self, pattern: str, path: Path, glob_filter: str) -> ToolResult:
        cmd = ["rg", "-n", "--max-count", "50", pattern, str(path)]
        if glob_filter:
            cmd.extend(["--glob", glob_filter])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return ToolResult(content="Search timed out.", is_error=True)

        if result.returncode == 1:
            return ToolResult(content=f"No matches for pattern '{pattern}'.")
        if result.returncode != 0:
            raise FileNotFoundError("rg not found")

        # Filter lines by security
        lines = []
        for line in result.stdout.splitlines()[:50]:
            parts = line.split(":", 1)
            if parts and self._security.is_path_allowed(parts[0]):
                lines.append(line)

        return ToolResult(content="\n".join(lines) if lines else f"No accessible matches for '{pattern}'.")

    def _python_search(self, pattern: str, path: Path, glob_filter: str) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(content=f"Invalid regex: {e}", is_error=True)

        results: list[str] = []
        file_pattern = glob_filter or "**/*"

        for fpath in path.glob(file_pattern):
            if not fpath.is_file() or not self._security.is_path_allowed(str(fpath)):
                continue
            try:
                with open(fpath, "r", errors="ignore") as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            rel = fpath.relative_to(self._working_dir) if fpath.is_relative_to(self._working_dir) else fpath
                            results.append(f"{rel}:{lineno}:{line.rstrip()}")
                            if len(results) >= 50:
                                break
            except (OSError, UnicodeDecodeError):
                continue
            if len(results) >= 50:
                break

        if not results:
            return ToolResult(content=f"No matches for pattern '{pattern}'.")
        return ToolResult(content="\n".join(results))


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a file with line numbers. Supports offset and limit for large files."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read."},
            "offset": {"type": "integer", "description": "Line number to start from (1-based).", "default": 1},
            "limit": {"type": "integer", "description": "Max lines to read.", "default": 200},
        },
        "required": ["path"],
    }

    def __init__(self, working_dir: str, security: SecurityPolicy):
        self._working_dir = Path(working_dir).resolve()
        self._security = security

    async def execute(self, path: str, offset: int = 1, limit: int = 200, **kwargs: Any) -> ToolResult:
        fpath = self._security.resolve_path(path)

        try:
            self._security.check_path(str(fpath))
        except Exception as e:
            return ToolResult(content=str(e), is_error=True)

        if not fpath.exists():
            return ToolResult(content=f"File not found: {path}", is_error=True)
        if not fpath.is_file():
            return ToolResult(content=f"Not a file: {path}", is_error=True)

        try:
            with open(fpath, "r", errors="replace") as f:
                all_lines = f.readlines()
        except OSError as e:
            return ToolResult(content=f"Error reading file: {e}", is_error=True)

        total = len(all_lines)
        start = max(0, offset - 1)
        end = min(total, start + limit)
        selected = all_lines[start:end]

        lines = []
        for i, line in enumerate(selected, start=start + 1):
            truncated = line.rstrip("\n")
            if len(truncated) > 2000:
                truncated = truncated[:2000] + "..."
            lines.append(f"{i:>6}\t{truncated}")

        header = f"File: {path} ({total} lines total, showing {start + 1}-{end})"
        return ToolResult(content=header + "\n" + "\n".join(lines))


class ListTreeTool(BaseTool):
    name = "list_tree"
    description = "Show directory tree structure."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path.", "default": "."},
            "max_depth": {"type": "integer", "description": "Maximum depth to traverse.", "default": 2},
        },
    }

    EXCLUDE_DIRS = {
        "__pycache__", ".git", "node_modules", ".tox", ".mypy_cache",
        ".pytest_cache", ".eggs", "*.egg-info", ".ipynb_checkpoints",
    }

    def __init__(self, working_dir: str, security: SecurityPolicy):
        self._working_dir = Path(working_dir).resolve()
        self._security = security

    async def execute(self, path: str = ".", max_depth: int = 2, **kwargs: Any) -> ToolResult:
        base = self._security.resolve_path(path)

        try:
            self._security.check_path(str(base))
        except SecurityError as e:
            return ToolResult(content=str(e), is_error=True)

        if not base.exists():
            return ToolResult(content=f"Directory not found: {path}", is_error=True)

        lines = [str(base.relative_to(self._working_dir) if base.is_relative_to(self._working_dir) else base) + "/"]
        self._walk(base, lines, prefix="", depth=0, max_depth=max_depth)

        if len(lines) > 200:
            lines = lines[:200]
            lines.append(f"... (truncated, {len(lines)} entries shown)")

        return ToolResult(content="\n".join(lines))

    def _walk(self, dir_path: Path, lines: list[str], prefix: str, depth: int, max_depth: int) -> None:
        if depth >= max_depth:
            return

        try:
            entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        entries = [e for e in entries if e.name not in self.EXCLUDE_DIRS and not e.name.startswith(".")]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            if not self._security.is_path_allowed(str(entry)):
                continue

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                self._walk(entry, lines, prefix + extension, depth + 1, max_depth)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")
