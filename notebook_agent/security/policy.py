"""Security policy enforcement for tool calls and code execution."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a security policy is violated."""


class SecurityPolicy:
    """Enforces security rules on tool calls and code execution.

    - Blocks forbidden code patterns (pip install, subprocess, etc.)
    - Blocks access to evaluator paths
    - Validates tool calls
    """

    def __init__(
        self,
        forbidden_code_patterns: list[str] | None = None,
        evaluator_paths: list[str] | None = None,
        working_dir: str = ".",
    ):
        self._working_dir = Path(working_dir).resolve()
        self._forbidden_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (forbidden_code_patterns or [])
        ]
        self._evaluator_paths = []
        for p in (evaluator_paths or []):
            ep = Path(p)
            if not ep.is_absolute():
                ep = self._working_dir / ep
            self._evaluator_paths.append(ep.resolve())

    def resolve_path(self, path: str) -> Path:
        """Resolve user-provided paths relative to the configured working dir."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._working_dir / candidate
        return candidate.resolve()

    def _ensure_inside_working_dir(self, resolved: Path, original_path: str) -> None:
        """Ensure path stays inside the configured workspace root."""
        try:
            resolved.relative_to(self._working_dir)
        except ValueError as exc:
            raise SecurityError(
                f"Access denied: '{original_path}' is outside the working directory "
                f"'{self._working_dir}'."
            ) from exc

    def check_code(self, code: str) -> None:
        """Check code before execution. Raises SecurityError if forbidden."""
        lowered = code.lower()
        for eval_path in self._evaluator_paths:
            full = str(eval_path).lower()
            name = eval_path.name.lower()
            if full and full in lowered:
                raise SecurityError("Forbidden code pattern detected: direct evaluator path access.")
            if name and name in lowered:
                raise SecurityError("Forbidden code pattern detected: evaluator file reference.")

        for pattern in self._forbidden_patterns:
            match = pattern.search(code)
            if match:
                raise SecurityError(
                    f"Forbidden code pattern detected: '{match.group()}'. "
                    f"This operation is not allowed."
                )

    def check_path(self, path: str) -> None:
        """Check if a file path is accessible. Raises SecurityError if blocked."""
        resolved = self.resolve_path(path)
        self._ensure_inside_working_dir(resolved, path)

        for eval_path in self._evaluator_paths:
            if eval_path.is_dir():
                # Block anything inside the evaluator directory
                try:
                    resolved.relative_to(eval_path)
                    raise SecurityError(
                        f"Access denied: '{path}' is inside the protected evaluator directory."
                    )
                except ValueError:
                    pass  # Not relative to eval_path, that's fine
            else:
                # Block the specific file
                if resolved == eval_path:
                    raise SecurityError(
                        f"Access denied: '{path}' is a protected evaluator file."
                    )

    def is_path_allowed(self, path: str) -> bool:
        """Check if a path is allowed (non-raising version)."""
        try:
            self.check_path(path)
            return True
        except SecurityError:
            return False

    def check_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Validate a tool call against security policy."""
        # Check file paths in tool arguments
        for key in ("path", "file_path"):
            if key in arguments:
                self.check_path(arguments[key])

        # Block code execution tools if code contains forbidden patterns
        if tool_name in ("run_cell", "run_from_cell", "run_stale_cells"):
            # Code is checked at execution time, not here
            pass
