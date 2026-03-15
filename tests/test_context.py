"""Tests for context renderer."""

from notebook_agent.context.renderer import ContextRenderer
from notebook_agent.models.cell import Cell, CellOutput, CellState, CellType
from notebook_agent.models.notebook import Notebook


def make_notebook_with_long_output() -> Notebook:
    long_output = "\n".join(f"line {i}" for i in range(100))
    return Notebook(cells=[
        Cell(
            id="c0", cell_type=CellType.CODE, source="print(range(100))",
            state=CellState.CLEAN, execution_count=1,
            outputs=[CellOutput(
                output_type="stream",
                content=long_output,
                full_content=long_output,
                line_count=100,
            )],
        ),
    ])


class TestContextRenderer:
    def test_empty_notebook(self):
        renderer = ContextRenderer()
        nb = Notebook()
        text = renderer.render(nb)
        assert "0 cells" in text
        assert "Empty" in text

    def test_basic_render(self, sample_notebook):
        renderer = ContextRenderer(unfold_last_n=2)
        text = renderer.render(sample_notebook)

        # Should have header
        assert "NOTEBOOK" in text
        assert "5 cells" in text

        # First cells should be folded
        assert "[0] MARKDOWN" in text
        assert "[1] CODE [CLEAN]" in text

        # Last cells should be unfolded (full source shown)
        assert "z = y + 1" in text  # cell 3 unfolded
        assert "## Summary" in text  # cell 4 unfolded

    def test_output_cropping(self):
        renderer = ContextRenderer(max_output_lines=10)
        nb = make_notebook_with_long_output()
        text = renderer.render(nb)

        assert "more lines" in text
        assert "expand_output(0)" in text

    def test_stale_annotation(self):
        nb = Notebook(cells=[
            Cell(id="c0", cell_type=CellType.CODE, source="x = 1",
                 state=CellState.STALE, execution_count=1),
        ])
        renderer = ContextRenderer()
        text = renderer.render(nb)
        assert "[STALE]" in text

    def test_dirty_annotation(self):
        nb = Notebook(cells=[
            Cell(id="c0", cell_type=CellType.CODE, source="x = 1",
                 state=CellState.DIRTY),
        ])
        renderer = ContextRenderer()
        text = renderer.render(nb)
        assert "[DIRTY]" in text

    def test_folded_cell_is_compact(self):
        nb = Notebook(cells=[
            Cell(id="c0", cell_type=CellType.CODE, source="import os\nimport sys\nimport math",
                 state=CellState.CLEAN, execution_count=1, folded=True),
            Cell(id="c1", cell_type=CellType.CODE, source="x = 1",
                 state=CellState.CLEAN, execution_count=2),
        ])
        renderer = ContextRenderer(unfold_last_n=5)  # Would normally unfold all
        text = renderer.render(nb)

        # Cell 0 should be folded despite unfold_last_n=5
        assert "[0] CODE [CLEAN]" in text
        # Should show first line and line count
        assert "import os" in text

    def test_explicit_unfold_override(self):
        nb = Notebook(cells=[
            Cell(id="c0", cell_type=CellType.CODE, source="x = 1",
                 state=CellState.CLEAN, execution_count=1, folded=False),
            Cell(id="c1", cell_type=CellType.CODE, source="y = x + 1",
                 state=CellState.CLEAN, execution_count=2),
            Cell(id="c2", cell_type=CellType.CODE, source="z = y + 1",
                 state=CellState.CLEAN, execution_count=3),
            Cell(id="c3", cell_type=CellType.CODE, source="w = z + 1",
                 state=CellState.CLEAN, execution_count=4),
        ])
        renderer = ContextRenderer(unfold_last_n=1)
        text = renderer.render(nb)
        # Cell 0 is in the folded section by default, but explicit unfold should show full block.
        assert "--- [0] CODE [CLEAN] exec:1 ---" in text

    def test_expand_cell(self, sample_notebook):
        renderer = ContextRenderer()
        text = renderer.expand_cell(sample_notebook, 1)
        assert "import math" in text
        assert "x = 42" in text

    def test_expand_output(self):
        nb = make_notebook_with_long_output()
        renderer = ContextRenderer()
        text = renderer.expand_output(nb, 0, 0)
        assert "line 99" in text  # Full content, not cropped

    def test_expand_output_invalid_index(self):
        nb = Notebook()
        renderer = ContextRenderer()
        text = renderer.expand_output(nb, 5)
        assert "Error" in text or "out of range" in text
