"""Notebook reading, writing, and execution tools."""

from __future__ import annotations

from typing import Any

from notebook_agent.context.renderer import ContextRenderer
from notebook_agent.kernel.manager import KernelManager
from notebook_agent.models.cell import CellType
from notebook_agent.notebook_ops.manager import NotebookManager
from notebook_agent.tools.base import BaseTool, ToolResult


class ReadNotebookTool(BaseTool):
    name = "read_notebook"
    description = "Render the current notebook. Use 'compact' for a brief overview or 'full' for all cells expanded."
    parameters = {
        "type": "object",
        "properties": {
            "detail": {
                "type": "string",
                "enum": ["compact", "full"],
                "description": "Level of detail: 'compact' (default) folds old cells, 'full' shows everything.",
                "default": "compact",
            }
        },
    }

    def __init__(self, nb_manager: NotebookManager, renderer: ContextRenderer, kernel: KernelManager):
        self._nb = nb_manager
        self._renderer = renderer
        self._kernel = kernel

    async def execute(self, detail: str = "compact", **kwargs: Any) -> ToolResult:
        if detail == "full":
            # Temporarily unfold all cells
            old_n = self._renderer.unfold_last_n
            try:
                self._renderer.unfold_last_n = len(self._nb.notebook.cells)
                text = self._renderer.render(self._nb.notebook, self._kernel.is_alive)
            finally:
                self._renderer.unfold_last_n = old_n
        else:
            text = self._renderer.render(self._nb.notebook, self._kernel.is_alive)
        return ToolResult(content=text)


class ReadCellTool(BaseTool):
    name = "read_cell"
    description = "Read a specific cell's full source code and output."
    parameters = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Cell index to read."},
        },
        "required": ["index"],
    }

    def __init__(self, nb_manager: NotebookManager, renderer: ContextRenderer):
        self._nb = nb_manager
        self._renderer = renderer

    async def execute(self, index: int, **kwargs: Any) -> ToolResult:
        text = self._renderer.expand_cell(self._nb.notebook, index)
        return ToolResult(content=text)


class ExpandOutputTool(BaseTool):
    name = "expand_output"
    description = "Show the full uncropped output of a cell."
    parameters = {
        "type": "object",
        "properties": {
            "cell_index": {"type": "integer", "description": "Cell index."},
            "output_index": {"type": "integer", "description": "Output index (default 0).", "default": 0},
        },
        "required": ["cell_index"],
    }

    def __init__(self, nb_manager: NotebookManager, renderer: ContextRenderer):
        self._nb = nb_manager
        self._renderer = renderer

    async def execute(self, cell_index: int, output_index: int = 0, **kwargs: Any) -> ToolResult:
        text = self._renderer.expand_output(self._nb.notebook, cell_index, output_index)
        return ToolResult(content=text)


class AddCellTool(BaseTool):
    name = "add_cell"
    description = "Insert a new cell. Use index=-1 to append at the end."
    parameters = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Position to insert (-1 for append).", "default": -1},
            "cell_type": {"type": "string", "enum": ["code", "markdown", "title"], "description": "Cell type."},
            "source": {"type": "string", "description": "Cell source content."},
        },
        "required": ["cell_type", "source"],
    }

    def __init__(self, nb_manager: NotebookManager):
        self._nb = nb_manager

    async def execute(self, cell_type: str, source: str, index: int = -1, **kwargs: Any) -> ToolResult:
        ct = CellType(cell_type)
        cell = self._nb.insert_cell(index, ct, source)
        actual_idx = self._nb.notebook.cells.index(cell)
        return ToolResult(content=f"Inserted {cell_type} cell at index [{actual_idx}].")


class EditCellTool(BaseTool):
    name = "edit_cell"
    description = (
        "Edit an existing cell's source. For code cells, this marks the cell as DIRTY "
        "and downstream cells as STALE. You should run the cell after editing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Cell index to edit."},
            "new_source": {"type": "string", "description": "New source content."},
        },
        "required": ["index", "new_source"],
    }

    def __init__(self, nb_manager: NotebookManager):
        self._nb = nb_manager

    async def execute(self, index: int, new_source: str, **kwargs: Any) -> ToolResult:
        try:
            cell = self._nb.edit_cell(index, new_source)
        except IndexError as e:
            return ToolResult(content=str(e), is_error=True)

        msg = f"Edited cell [{index}] ({cell.cell_type.value})."
        if cell.is_code:
            stale_indices = self._nb.notebook.downstream_code_indices(index)
            stale_cells = [i for i in stale_indices if self._nb.notebook.cells[i].state.value == "stale"]
            if stale_cells:
                msg += f" Downstream cells {stale_cells} are now STALE."
            msg += " Run the cell to update outputs."
        return ToolResult(content=msg)


class DeleteCellTool(BaseTool):
    name = "delete_cell"
    description = (
        "Delete a cell. For code cells, downstream cells become STALE. "
        "Consider inserting a summary markdown cell to preserve context."
    )
    parameters = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Cell index to delete."},
        },
        "required": ["index"],
    }

    def __init__(self, nb_manager: NotebookManager):
        self._nb = nb_manager

    async def execute(self, index: int, **kwargs: Any) -> ToolResult:
        try:
            cell = self._nb.delete_cell(index)
        except IndexError as e:
            return ToolResult(content=str(e), is_error=True)

        msg = f"Deleted cell [{index}] ({cell.cell_type.value})."
        if cell.is_code:
            msg += (
                " Downstream code cells may now be STALE. "
                "Consider adding a summary markdown cell to preserve important context."
            )
        return ToolResult(content=msg)


class RunCellTool(BaseTool):
    name = "run_cell"
    description = "Execute a single code cell and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Cell index to execute."},
        },
        "required": ["index"],
    }

    def __init__(self, nb_manager: NotebookManager, security_policy=None):
        self._nb = nb_manager
        self._security = security_policy

    async def execute(self, index: int, **kwargs: Any) -> ToolResult:
        try:
            cell = self._nb.notebook.cells[index]
        except IndexError:
            return ToolResult(content=f"Cell index {index} out of range.", is_error=True)

        if not cell.is_code:
            return ToolResult(content=f"Cell [{index}] is a markdown cell, nothing to execute.")

        # Security check on the code
        if self._security:
            from notebook_agent.security.policy import SecurityError
            try:
                self._security.check_code(cell.source)
            except SecurityError as e:
                return ToolResult(content=f"Security error: {e}", is_error=True)

        result = await self._nb.run_cell(index)

        parts = [f"Cell [{index}] executed (status: {result.status})."]
        if result.execution_count:
            parts[0] = f"Cell [{index}] executed (exec:{result.execution_count}, status: {result.status})."

        for out in result.outputs:
            if out.content:
                parts.append(out.content)

        if result.error:
            parts.append(f"\nError:\n{result.error}")

        return ToolResult(content="\n".join(parts), is_error=result.status == "error")


class RunFromCellTool(BaseTool):
    name = "run_from_cell"
    description = "Run a cell and all code cells below it. Stops on first error."
    parameters = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Starting cell index."},
        },
        "required": ["index"],
    }

    def __init__(self, nb_manager: NotebookManager, security_policy=None):
        self._nb = nb_manager
        self._security = security_policy

    async def execute(self, index: int, **kwargs: Any) -> ToolResult:
        # Security check on all code cells that will run
        if self._security:
            from notebook_agent.security.policy import SecurityError
            indices = [index] + self._nb.notebook.downstream_code_indices(index)
            for i in indices:
                if i < len(self._nb.notebook.cells) and self._nb.notebook.cells[i].is_code:
                    try:
                        self._security.check_code(self._nb.notebook.cells[i].source)
                    except SecurityError as e:
                        return ToolResult(content=f"Security error in cell [{i}]: {e}", is_error=True)

        results = await self._nb.run_from(index)
        parts = []
        for r in results:
            status = f"exec:{r.execution_count}" if r.execution_count else r.status
            parts.append(f"[{status}] {r.status}")
            for out in r.outputs:
                if out.content:
                    # Truncate long outputs
                    lines = out.content.splitlines()
                    if len(lines) > 20:
                        parts.append("\n".join(lines[:10]))
                        parts.append(f"... [{len(lines) - 20} more lines]")
                        parts.append("\n".join(lines[-10:]))
                    else:
                        parts.append(out.content)

        return ToolResult(content="\n".join(parts), is_error=any(r.status == "error" for r in results))


class RunStaleCellsTool(BaseTool):
    name = "run_stale_cells"
    description = "Run all DIRTY and STALE code cells in order. Stops on first error."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, nb_manager: NotebookManager, security_policy=None):
        self._nb = nb_manager
        self._security = security_policy

    async def execute(self, **kwargs: Any) -> ToolResult:
        # Security check
        if self._security:
            from notebook_agent.security.policy import SecurityError
            for i, cell in enumerate(self._nb.notebook.cells):
                if cell.is_code and cell.state.value in ("dirty", "stale"):
                    try:
                        self._security.check_code(cell.source)
                    except SecurityError as e:
                        return ToolResult(content=f"Security error in cell [{i}]: {e}", is_error=True)

        results = await self._nb.run_stale()
        if not results:
            return ToolResult(content="No dirty or stale cells to run.")

        parts = [f"Ran {len(results)} cell(s)."]
        for r in results:
            parts.append(f"  exec:{r.execution_count} - {r.status}")
            if r.error:
                parts.append(f"    Error: {r.error[:200]}")

        return ToolResult(content="\n".join(parts), is_error=any(r.status == "error" for r in results))


class RestartKernelTool(BaseTool):
    name = "restart_kernel"
    description = "Restart the Jupyter kernel. WARNING: This clears all execution state. All cells become UNEXECUTED."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, kernel: KernelManager, nb_manager: NotebookManager):
        self._kernel = kernel
        self._nb = nb_manager

    async def execute(self, **kwargs: Any) -> ToolResult:
        await self._kernel.restart()
        # Mark all code cells as unexecuted
        from notebook_agent.models.cell import CellState
        for cell in self._nb.notebook.cells:
            if cell.is_code:
                cell.state = CellState.UNEXECUTED
                cell.execution_count = None
                cell.outputs = []
        self._nb._bump()
        return ToolResult(content="Kernel restarted. All code cells are now UNEXECUTED.")


class CreateSavepointTool(BaseTool):
    name = "create_savepoint"
    description = "Create a named savepoint (snapshot) of the current notebook state."
    parameters = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "Optional label for this savepoint.", "default": ""},
        },
    }

    def __init__(self, nb_manager: NotebookManager):
        self._nb = nb_manager

    async def execute(self, label: str = "", **kwargs: Any) -> ToolResult:
        sp_id = self._nb.create_savepoint(label)
        return ToolResult(content=f"Savepoint created: {sp_id} ({label or 'unlabeled'})")


class ListSavepointsTool(BaseTool):
    name = "list_savepoints"
    description = "List available savepoints."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, nb_manager: NotebookManager):
        self._nb = nb_manager

    async def execute(self, **kwargs: Any) -> ToolResult:
        savepoints = self._nb.list_savepoints()
        if not savepoints:
            return ToolResult(content="No savepoints available.")
        lines = []
        for sp in savepoints:
            lines.append(
                f"{sp['id']} | {sp['label']} | cells:{sp['cell_count']} | ts:{sp['timestamp']:.0f}"
            )
        return ToolResult(content="\n".join(lines))


class RestoreSavepointTool(BaseTool):
    name = "restore_savepoint"
    description = "Restore notebook to a previous savepoint by ID."
    parameters = {
        "type": "object",
        "properties": {
            "savepoint_id": {"type": "string", "description": "Savepoint ID to restore."},
        },
        "required": ["savepoint_id"],
    }

    def __init__(self, nb_manager: NotebookManager):
        self._nb = nb_manager

    async def execute(self, savepoint_id: str, **kwargs: Any) -> ToolResult:
        try:
            self._nb.restore_savepoint(savepoint_id)
        except ValueError as e:
            return ToolResult(content=str(e), is_error=True)
        return ToolResult(content=f"Restored savepoint: {savepoint_id}")
