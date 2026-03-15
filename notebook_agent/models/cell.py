"""Cell and CellOutput data models for the notebook."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CellType(str, Enum):
    CODE = "code"
    MARKDOWN = "markdown"
    TITLE = "title"


class CellState(str, Enum):
    CLEAN = "clean"          # Executed, output matches source
    DIRTY = "dirty"          # Source edited since last execution
    STALE = "stale"          # An upstream cell was edited/deleted
    UNEXECUTED = "unexecuted"  # Never run


@dataclass
class CellOutput:
    """A single output from cell execution."""

    output_type: str          # "stream", "execute_result", "display_data", "error"
    content: str              # Text content for LLM display
    raw: dict = field(default_factory=dict)  # Original nbformat output dict
    line_count: int = 0
    full_content: str = ""    # Uncropped full content

    @property
    def cropped(self) -> bool:
        return self.content != self.full_content

    @staticmethod
    def from_nbformat(output: dict) -> CellOutput:
        """Parse an nbformat output dict into a CellOutput."""
        output_type = output.get("output_type", "unknown")
        text = ""

        if output_type == "stream":
            text = output.get("text", "")
        elif output_type in ("execute_result", "display_data"):
            data = output.get("data", {})
            text = data.get("text/plain", "")
            if isinstance(text, list):
                text = "".join(text)
        elif output_type == "error":
            ename = output.get("ename", "Error")
            evalue = output.get("evalue", "")
            traceback_lines = output.get("traceback", [])
            # Strip ANSI codes from traceback
            import re
            ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
            cleaned = [ansi_escape.sub("", line) for line in traceback_lines]
            text = f"{ename}: {evalue}\n" + "\n".join(cleaned)

        if isinstance(text, list):
            text = "".join(text)

        line_count = len(text.splitlines()) if text else 0

        return CellOutput(
            output_type=output_type,
            content=text,
            raw=output,
            line_count=line_count,
            full_content=text,
        )


@dataclass
class Cell:
    """A single notebook cell."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cell_type: CellType = CellType.CODE
    source: str = ""
    outputs: list[CellOutput] = field(default_factory=list)
    state: CellState = CellState.UNEXECUTED
    execution_count: int | None = None
    # None -> renderer default behavior; True -> force folded; False -> force unfolded
    folded: bool | None = None
    last_executed_source: str | None = None
    stale_reason: str | None = None

    @property
    def first_line(self) -> str:
        lines = self.source.strip().splitlines()
        return lines[0] if lines else ""

    @property
    def line_count(self) -> int:
        return len(self.source.splitlines())

    @property
    def is_code(self) -> bool:
        return self.cell_type == CellType.CODE

    @property
    def is_markdown(self) -> bool:
        return self.cell_type == CellType.MARKDOWN

    @property
    def is_title(self) -> bool:
        return self.cell_type == CellType.TITLE
