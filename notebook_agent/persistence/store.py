"""Notebook <-> .ipynb file serialization."""

from __future__ import annotations

from pathlib import Path

import nbformat

from notebook_agent.models.cell import Cell, CellOutput, CellState, CellType
from notebook_agent.models.notebook import Notebook


class NotebookStore:
    """Serialize/deserialize Notebook to .ipynb (nbformat v4)."""

    @staticmethod
    def save(notebook: Notebook, path: str | Path) -> None:
        """Save notebook to .ipynb format.

        Custom metadata (state, folded, stale_reason) is stored in cell metadata
        under the 'notebook_agent' key.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        nb = nbformat.v4.new_notebook()
        nb.metadata = notebook.metadata.copy()
        nb.metadata["notebook_agent_version"] = notebook.version

        for cell in notebook.cells:
            if cell.cell_type == CellType.CODE:
                nb_cell = nbformat.v4.new_code_cell(source=cell.source)
                nb_cell.execution_count = cell.execution_count
                nb_cell.outputs = [
                    nbformat.NotebookNode(out.raw) for out in cell.outputs if out.raw
                ]
            else:
                # Title cells are persisted as markdown with a custom subtype marker.
                nb_cell = nbformat.v4.new_markdown_cell(source=cell.source)

            nb_cell.id = cell.id
            nb_cell.metadata["notebook_agent"] = {
                "cell_type": cell.cell_type.value,
                "state": cell.state.value,
                "folded": cell.folded,
                "stale_reason": cell.stale_reason,
                "last_executed_source": cell.last_executed_source,
            }
            nb.cells.append(nb_cell)

        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    @staticmethod
    def load(path: str | Path) -> Notebook:
        """Load notebook from .ipynb.

        When a code cell has stored outputs, it is loaded as STALE because
        kernel memory is not persisted across process restarts.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        cells: list[Cell] = []
        for nb_cell in nb.cells:
            # Read custom metadata if present
            agent_meta = nb_cell.metadata.get("notebook_agent", {})
            cell_type_raw = agent_meta.get("cell_type")
            try:
                if cell_type_raw is not None:
                    cell_type = CellType(cell_type_raw)
                elif nb_cell.cell_type == "code":
                    cell_type = CellType.CODE
                else:
                    cell_type = CellType.MARKDOWN
            except ValueError:
                cell_type = CellType.MARKDOWN

            outputs: list[CellOutput] = []
            if cell_type == CellType.CODE:
                for out in nb_cell.get("outputs", []):
                    outputs.append(CellOutput.from_nbformat(out))

            if cell_type == CellType.CODE:
                if outputs:
                    # Keep old outputs visible but clearly stale after reload.
                    state = CellState.STALE
                    stale_reason = (
                        agent_meta.get("stale_reason")
                        or "loaded from disk; kernel state is from an older environment"
                    )
                else:
                    saved_state = agent_meta.get("state")
                    if saved_state == CellState.DIRTY.value:
                        state = CellState.DIRTY
                    else:
                        state = CellState.UNEXECUTED
                    stale_reason = agent_meta.get("stale_reason")
            else:
                state = CellState.CLEAN
                stale_reason = None

            cell = Cell(
                id=nb_cell.get("id", None) or Cell().id,
                cell_type=cell_type,
                source=nb_cell.source,
                outputs=outputs,
                state=state,
                execution_count=nb_cell.get("execution_count"),
                folded=agent_meta.get("folded", None),
                last_executed_source=agent_meta.get("last_executed_source"),
                stale_reason=stale_reason,
            )
            cells.append(cell)

        metadata = dict(nb.metadata)
        version = metadata.pop("notebook_agent_version", 0)

        return Notebook(cells=cells, metadata=metadata, version=version)

    @staticmethod
    def create_empty(path: str | Path) -> Notebook:
        """Create and save an empty notebook."""
        notebook = Notebook()
        NotebookStore.save(notebook, path)
        return notebook
