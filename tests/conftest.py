"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from notebook_agent.config import AgentConfig
from notebook_agent.models.cell import Cell, CellOutput, CellState, CellType
from notebook_agent.models.notebook import Notebook


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_notebook() -> Notebook:
    """Create a sample notebook with mixed cell types and states."""
    cells = [
        Cell(
            id="cell-0",
            cell_type=CellType.MARKDOWN,
            source="# Test Notebook",
            state=CellState.CLEAN,
        ),
        Cell(
            id="cell-1",
            cell_type=CellType.CODE,
            source="import math\nx = 42",
            state=CellState.CLEAN,
            execution_count=1,
            last_executed_source="import math\nx = 42",
            outputs=[
                CellOutput(
                    output_type="execute_result",
                    content="42",
                    raw={"output_type": "execute_result", "data": {"text/plain": "42"}, "metadata": {}, "execution_count": 1},
                    line_count=1,
                    full_content="42",
                )
            ],
        ),
        Cell(
            id="cell-2",
            cell_type=CellType.CODE,
            source="y = x * 2\nprint(y)",
            state=CellState.CLEAN,
            execution_count=2,
            last_executed_source="y = x * 2\nprint(y)",
            outputs=[
                CellOutput(
                    output_type="stream",
                    content="84\n",
                    raw={"output_type": "stream", "name": "stdout", "text": "84\n"},
                    line_count=1,
                    full_content="84\n",
                )
            ],
        ),
        Cell(
            id="cell-3",
            cell_type=CellType.CODE,
            source="z = y + 1",
            state=CellState.UNEXECUTED,
        ),
        Cell(
            id="cell-4",
            cell_type=CellType.MARKDOWN,
            source="## Summary\nDone with calculations.",
        ),
    ]
    return Notebook(cells=cells, version=3)


@pytest.fixture
def config(tmp_dir) -> AgentConfig:
    """Create a test configuration."""
    return AgentConfig(
        notebook_path=str(tmp_dir / "test.ipynb"),
        working_dir=str(tmp_dir),
        checkpoint_dir=str(tmp_dir / "checkpoints"),
        auto_save=False,
    )
