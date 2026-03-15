"""Tests for staleness tracking."""

from notebook_agent.models.cell import Cell, CellState, CellType
from notebook_agent.models.notebook import Notebook
from notebook_agent.notebook_ops.staleness import StalenessTracker


def make_notebook() -> Notebook:
    """Create a notebook with 5 code cells, all CLEAN."""
    cells = [
        Cell(id=f"c{i}", cell_type=CellType.CODE, source=f"x{i} = {i}", state=CellState.CLEAN)
        for i in range(5)
    ]
    # Insert a markdown cell at index 2
    cells.insert(2, Cell(id="md", cell_type=CellType.MARKDOWN, source="# Note"))
    return Notebook(cells=cells)


class TestStalenessTracker:
    def test_edit_code_cell_marks_dirty_and_downstream_stale(self):
        nb = make_notebook()
        # Cells: [code0, code1, markdown2, code3, code4, code5]
        changed = StalenessTracker.on_edit_code_cell(nb, 1)

        assert nb.cells[1].state == CellState.DIRTY
        assert 1 in changed

        # Downstream code cells (3, 4, 5) should be STALE
        assert nb.cells[3].state == CellState.STALE
        assert nb.cells[4].state == CellState.STALE
        assert nb.cells[5].state == CellState.STALE

        # Markdown cell should be unaffected
        assert nb.cells[2].cell_type == CellType.MARKDOWN

    def test_edit_code_cell_preserves_already_stale(self):
        nb = make_notebook()
        nb.cells[4].state = CellState.STALE

        changed = StalenessTracker.on_edit_code_cell(nb, 1)
        # Cell 4 was already STALE, shouldn't be in changed
        assert 4 not in changed

    def test_delete_code_cell_marks_downstream_stale(self):
        nb = make_notebook()
        changed = StalenessTracker.on_delete_code_cell(nb, 0)

        assert nb.cells[1].state == CellState.STALE
        assert nb.cells[3].state == CellState.STALE

    def test_execute_cell_marks_clean(self):
        nb = make_notebook()
        nb.cells[1].state = CellState.DIRTY

        changed = StalenessTracker.on_execute_cell(nb, 1)
        assert nb.cells[1].state == CellState.CLEAN
        assert 1 in changed

    def test_edit_markdown_is_noop(self):
        nb = make_notebook()
        changed = StalenessTracker.on_edit_markdown_cell(nb, 2)
        assert changed == []
        # No cells should have changed state
        for cell in nb.cells:
            if cell.is_code:
                assert cell.state == CellState.CLEAN
