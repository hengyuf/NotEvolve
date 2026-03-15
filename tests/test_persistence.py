"""Tests for notebook persistence."""

from pathlib import Path

from notebook_agent.config import AgentConfig
from notebook_agent.models.cell import Cell, CellOutput, CellState, CellType
from notebook_agent.models.notebook import Notebook
from notebook_agent.notebook_ops.manager import NotebookManager
from notebook_agent.persistence.store import NotebookStore


class TestNotebookStore:
    def test_save_and_load_empty(self, tmp_path):
        path = tmp_path / "empty.ipynb"
        nb = Notebook()
        NotebookStore.save(nb, path)

        loaded = NotebookStore.load(path)
        assert len(loaded) == 0

    def test_save_and_load_with_cells(self, tmp_path):
        path = tmp_path / "test.ipynb"
        nb = Notebook(
            cells=[
                Cell(id="c1", cell_type=CellType.CODE, source="x = 1", state=CellState.CLEAN),
                Cell(id="c2", cell_type=CellType.MARKDOWN, source="# Hello"),
            ],
            version=5,
        )
        NotebookStore.save(nb, path)

        loaded = NotebookStore.load(path)
        assert len(loaded) == 2
        assert loaded[0].cell_type == CellType.CODE
        assert loaded[0].source == "x = 1"
        assert loaded[1].cell_type == CellType.MARKDOWN
        assert loaded[1].source == "# Hello"
        assert loaded.version == 5

    def test_loaded_cells_are_unexecuted(self, tmp_path):
        path = tmp_path / "test.ipynb"
        nb = Notebook(
            cells=[
                Cell(id="c1", cell_type=CellType.CODE, source="x = 1", state=CellState.CLEAN,
                     execution_count=1),
            ],
        )
        NotebookStore.save(nb, path)

        loaded = NotebookStore.load(path)
        assert loaded[0].state == CellState.UNEXECUTED

    def test_loaded_cells_with_outputs_are_stale(self, tmp_path):
        path = tmp_path / "with_output.ipynb"
        raw_out = {"output_type": "stream", "name": "stdout", "text": "hello\n"}
        nb = Notebook(
            cells=[
                Cell(
                    id="c1",
                    cell_type=CellType.CODE,
                    source="print('hello')",
                    outputs=[CellOutput.from_nbformat(raw_out)],
                    state=CellState.CLEAN,
                )
            ]
        )
        NotebookStore.save(nb, path)

        loaded = NotebookStore.load(path)
        assert loaded[0].state == CellState.STALE
        assert loaded[0].stale_reason is not None

    def test_preserves_outputs(self, tmp_path):
        path = tmp_path / "test.ipynb"
        raw_out = {"output_type": "stream", "name": "stdout", "text": "hello\n"}
        nb = Notebook(
            cells=[
                Cell(
                    id="c1", cell_type=CellType.CODE, source="print('hello')",
                    outputs=[CellOutput.from_nbformat(raw_out)],
                    state=CellState.CLEAN,
                ),
            ],
        )
        NotebookStore.save(nb, path)

        loaded = NotebookStore.load(path)
        assert len(loaded[0].outputs) == 1
        assert "hello" in loaded[0].outputs[0].content

    def test_preserves_custom_metadata(self, tmp_path):
        path = tmp_path / "test.ipynb"
        nb = Notebook(
            cells=[
                Cell(
                    id="c1", cell_type=CellType.CODE, source="x = 1",
                    state=CellState.DIRTY,
                    folded=True,
                    stale_reason="upstream edited",
                    last_executed_source="x = 0",
                ),
            ],
        )
        NotebookStore.save(nb, path)

        loaded = NotebookStore.load(path)
        # State resets to UNEXECUTED on load
        assert loaded[0].state == CellState.UNEXECUTED
        # But custom metadata is preserved
        assert loaded[0].folded is True
        assert loaded[0].stale_reason == "upstream edited"
        assert loaded[0].last_executed_source == "x = 0"

    def test_create_empty(self, tmp_path):
        path = tmp_path / "new.ipynb"
        nb = NotebookStore.create_empty(path)
        assert len(nb) == 0
        assert path.exists()

    def test_title_cells_roundtrip(self, tmp_path):
        path = tmp_path / "title.ipynb"
        nb = Notebook(
            cells=[Cell(id="t1", cell_type=CellType.TITLE, source="# Heading")],
        )
        NotebookStore.save(nb, path)
        loaded = NotebookStore.load(path)
        assert loaded[0].cell_type == CellType.TITLE
        assert loaded[0].source == "# Heading"

    def test_savepoints_persist_on_disk(self, tmp_path):
        class DummyExecutor:
            async def execute(self, code: str, timeout: int = 120):  # pragma: no cover - not used
                raise RuntimeError("not used in this test")

        config = AgentConfig(
            notebook_path=str(tmp_path / "nb.ipynb"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            auto_save=False,
        )
        nb = Notebook(cells=[Cell(id="c1", cell_type=CellType.CODE, source="x=1")])
        manager = NotebookManager(nb, DummyExecutor(), config)

        sp_id = manager.create_savepoint("first")
        manager.insert_cell(-1, CellType.MARKDOWN, "note")
        manager.restore_savepoint(sp_id)

        assert len(manager.notebook.cells) == 1
        assert manager.notebook.cells[0].source == "x=1"

        manager2 = NotebookManager(Notebook(), DummyExecutor(), config)
        savepoints = manager2.list_savepoints()
        assert any(sp["id"] == sp_id for sp in savepoints)
