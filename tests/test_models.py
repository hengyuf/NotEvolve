"""Tests for core data models."""

from notebook_agent.models.cell import Cell, CellOutput, CellState, CellType
from notebook_agent.models.notebook import Notebook


class TestCellOutput:
    def test_from_nbformat_stream(self):
        raw = {"output_type": "stream", "name": "stdout", "text": "hello world\n"}
        out = CellOutput.from_nbformat(raw)
        assert out.output_type == "stream"
        assert out.content == "hello world\n"
        assert out.line_count == 1
        assert not out.cropped

    def test_from_nbformat_execute_result(self):
        raw = {
            "output_type": "execute_result",
            "data": {"text/plain": "42"},
            "metadata": {},
            "execution_count": 1,
        }
        out = CellOutput.from_nbformat(raw)
        assert out.output_type == "execute_result"
        assert out.content == "42"

    def test_from_nbformat_error(self):
        raw = {
            "output_type": "error",
            "ename": "ValueError",
            "evalue": "bad value",
            "traceback": ["ValueError: bad value"],
        }
        out = CellOutput.from_nbformat(raw)
        assert out.output_type == "error"
        assert "ValueError" in out.content
        assert "bad value" in out.content

    def test_cropped_property(self):
        out = CellOutput(
            output_type="stream",
            content="short",
            full_content="short version with more text",
            line_count=1,
        )
        assert out.cropped

        out2 = CellOutput(
            output_type="stream",
            content="same",
            full_content="same",
            line_count=1,
        )
        assert not out2.cropped


class TestCell:
    def test_defaults(self):
        cell = Cell()
        assert cell.cell_type == CellType.CODE
        assert cell.state == CellState.UNEXECUTED
        assert cell.source == ""
        assert cell.id  # UUID generated

    def test_first_line(self):
        cell = Cell(source="import math\nx = 42")
        assert cell.first_line == "import math"

    def test_line_count(self):
        cell = Cell(source="a\nb\nc")
        assert cell.line_count == 3

    def test_is_code_markdown(self):
        code = Cell(cell_type=CellType.CODE)
        md = Cell(cell_type=CellType.MARKDOWN)
        title = Cell(cell_type=CellType.TITLE)
        assert code.is_code
        assert not code.is_markdown
        assert md.is_markdown
        assert not md.is_code
        assert title.is_title


class TestNotebook:
    def test_empty(self):
        nb = Notebook()
        assert len(nb) == 0
        assert nb.version == 0

    def test_bump_version(self):
        nb = Notebook()
        assert nb.bump_version() == 1
        assert nb.bump_version() == 2

    def test_code_cell_indices(self, sample_notebook):
        indices = sample_notebook.code_cell_indices()
        assert indices == [1, 2, 3]

    def test_downstream_code_indices(self, sample_notebook):
        assert sample_notebook.downstream_code_indices(1) == [2, 3]
        assert sample_notebook.downstream_code_indices(2) == [3]
        assert sample_notebook.downstream_code_indices(3) == []
