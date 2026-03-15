"""Context management tools for folding/unfolding cells."""

from __future__ import annotations

from typing import Any

from notebook_agent.notebook_ops.manager import NotebookManager
from notebook_agent.tools.base import BaseTool, ToolResult


class FoldCellTool(BaseTool):
    name = "fold_cell"
    description = "Fold (collapse) a cell so it shows as a compact one-line summary in the notebook view."
    parameters = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Cell index to fold."},
        },
        "required": ["index"],
    }

    def __init__(self, nb_manager: NotebookManager):
        self._nb = nb_manager

    async def execute(self, index: int, **kwargs: Any) -> ToolResult:
        if index < 0 or index >= len(self._nb.notebook.cells):
            return ToolResult(content=f"Cell index {index} out of range.", is_error=True)

        self._nb.notebook.cells[index].folded = True
        self._nb._bump()
        return ToolResult(content=f"Cell [{index}] folded.")


class UnfoldCellTool(BaseTool):
    name = "unfold_cell"
    description = "Unfold (expand) a cell to show its full source and output in the notebook view."
    parameters = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Cell index to unfold."},
        },
        "required": ["index"],
    }

    def __init__(self, nb_manager: NotebookManager):
        self._nb = nb_manager

    async def execute(self, index: int, **kwargs: Any) -> ToolResult:
        if index < 0 or index >= len(self._nb.notebook.cells):
            return ToolResult(content=f"Cell index {index} out of range.", is_error=True)

        self._nb.notebook.cells[index].folded = False
        self._nb._bump()
        return ToolResult(content=f"Cell [{index}] unfolded.")
