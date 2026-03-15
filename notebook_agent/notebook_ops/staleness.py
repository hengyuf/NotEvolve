"""Linear staleness propagation for notebook cells."""

from __future__ import annotations

from notebook_agent.models.cell import CellState
from notebook_agent.models.notebook import Notebook


class StalenessTracker:
    """Propagates staleness through the notebook's code cells.

    Staleness is linear: editing/deleting a code cell marks all downstream
    code cells as STALE. No dependency analysis is performed.
    """

    @staticmethod
    def on_edit_code_cell(notebook: Notebook, index: int) -> list[int]:
        """Mark cell at index as DIRTY, downstream code cells as STALE.

        Returns list of indices that changed state.
        """
        changed: list[int] = []
        cell = notebook.cells[index]

        if cell.state != CellState.DIRTY:
            cell.state = CellState.DIRTY
            cell.stale_reason = None
            changed.append(index)

        for i in notebook.downstream_code_indices(index):
            downstream = notebook.cells[i]
            if downstream.state not in (CellState.STALE, CellState.UNEXECUTED):
                downstream.state = CellState.STALE
                downstream.stale_reason = f"upstream cell [{index}] was edited"
                changed.append(i)

        return changed

    @staticmethod
    def on_delete_code_cell(notebook: Notebook, deleted_index: int) -> list[int]:
        """Mark downstream code cells as STALE after a deletion.

        Called BEFORE the cell is actually removed from the list,
        so indices are still valid.

        Returns list of indices that changed state.
        """
        changed: list[int] = []

        for i in notebook.downstream_code_indices(deleted_index):
            downstream = notebook.cells[i]
            if downstream.state not in (CellState.STALE, CellState.UNEXECUTED):
                downstream.state = CellState.STALE
                downstream.stale_reason = f"upstream cell [{deleted_index}] was deleted"
                changed.append(i)

        return changed

    @staticmethod
    def on_execute_cell(notebook: Notebook, index: int) -> list[int]:
        """Mark cell as CLEAN after successful execution.

        Returns list of indices that changed state.
        """
        changed: list[int] = []
        cell = notebook.cells[index]

        if cell.state != CellState.CLEAN:
            cell.state = CellState.CLEAN
            cell.stale_reason = None
            cell.last_executed_source = cell.source
            changed.append(index)

        return changed

    @staticmethod
    def on_edit_markdown_cell(notebook: Notebook, index: int) -> list[int]:
        """No-op for staleness. Markdown edits don't affect execution.

        Returns empty list.
        """
        return []
