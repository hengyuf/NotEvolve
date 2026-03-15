"""NotebookManager: all notebook mutations, execution, and savepoints."""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from notebook_agent.config import AgentConfig
from notebook_agent.kernel.executor import CellExecutor, ExecutionResult
from notebook_agent.models.cell import Cell, CellState, CellType
from notebook_agent.models.notebook import Notebook
from notebook_agent.notebook_ops.staleness import StalenessTracker
from notebook_agent.persistence.store import NotebookStore

logger = logging.getLogger(__name__)


@dataclass
class Savepoint:
    """A snapshot of the notebook at a point in time."""

    id: str
    label: str
    timestamp: float
    snapshot_path: str
    cell_count: int


class NotebookManager:
    """Central manager for all notebook operations.

    All mutations go through this class to maintain staleness invariants,
    auto-save, and history tracking.
    """

    def __init__(
        self,
        notebook: Notebook,
        executor: CellExecutor,
        config: AgentConfig,
    ):
        self.notebook = notebook
        self._executor = executor
        self._config = config
        self._staleness = StalenessTracker()
        self._savepoint_dir = Path(self._config.checkpoint_dir) / "savepoints"
        self._savepoint_dir.mkdir(parents=True, exist_ok=True)
        self._savepoint_index_path = self._savepoint_dir / "index.json"
        self._savepoints: list[Savepoint] = self._load_savepoints()

    def _auto_save(self) -> None:
        """Save notebook to disk if auto_save is enabled."""
        if self._config.auto_save:
            NotebookStore.save(self.notebook, self._config.notebook_path)

    def _bump(self) -> None:
        """Increment version and auto-save."""
        self.notebook.bump_version()
        self._auto_save()

    def _load_savepoints(self) -> list[Savepoint]:
        """Load persistent savepoint metadata from disk."""
        if not self._savepoint_index_path.exists():
            return []
        try:
            with open(self._savepoint_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to load savepoint index: %s", self._savepoint_index_path)
            return []

        savepoints: list[Savepoint] = []
        for entry in data:
            try:
                savepoints.append(
                    Savepoint(
                        id=entry["id"],
                        label=entry["label"],
                        timestamp=float(entry["timestamp"]),
                        snapshot_path=entry["snapshot_path"],
                        cell_count=int(entry.get("cell_count", 0)),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        return savepoints

    def _persist_savepoints(self) -> None:
        """Persist savepoint metadata to disk."""
        data = [
            {
                "id": sp.id,
                "label": sp.label,
                "timestamp": sp.timestamp,
                "snapshot_path": sp.snapshot_path,
                "cell_count": sp.cell_count,
            }
            for sp in self._savepoints
        ]
        with open(self._savepoint_index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # --- CRUD Operations ---

    def insert_cell(
        self,
        index: int,
        cell_type: CellType,
        source: str,
    ) -> Cell:
        """Insert a new cell at the given index. Use -1 to append."""
        cell = Cell(
            id=str(uuid.uuid4()),
            cell_type=cell_type,
            source=source,
            state=CellState.UNEXECUTED if cell_type == CellType.CODE else CellState.CLEAN,
        )

        if index == -1 or index >= len(self.notebook.cells):
            self.notebook.cells.append(cell)
        else:
            self.notebook.cells.insert(index, cell)

        self._bump()
        logger.info("Inserted %s cell at index %d", cell_type.value, index)
        return cell

    def edit_cell(self, index: int, new_source: str) -> Cell:
        """Edit cell source and propagate staleness."""
        if index < 0 or index >= len(self.notebook.cells):
            raise IndexError(f"Cell index {index} out of range (0-{len(self.notebook.cells) - 1})")

        cell = self.notebook.cells[index]
        old_source = cell.source
        cell.source = new_source

        if cell.is_code:
            self._staleness.on_edit_code_cell(self.notebook, index)
        else:
            self._staleness.on_edit_markdown_cell(self.notebook, index)

        self._bump()
        logger.info("Edited cell [%d] (%s)", index, cell.cell_type.value)
        return cell

    def delete_cell(self, index: int) -> Cell:
        """Delete a cell and propagate staleness."""
        if index < 0 or index >= len(self.notebook.cells):
            raise IndexError(f"Cell index {index} out of range (0-{len(self.notebook.cells) - 1})")

        cell = self.notebook.cells[index]

        if cell.is_code:
            self._staleness.on_delete_code_cell(self.notebook, index)

        removed = self.notebook.cells.pop(index)
        self._bump()
        logger.info("Deleted cell [%d] (%s)", index, cell.cell_type.value)
        return removed

    # --- Execution ---

    async def run_cell(self, index: int, timeout: int | None = None) -> ExecutionResult:
        """Execute a single cell and update its state and outputs."""
        if index < 0 or index >= len(self.notebook.cells):
            raise IndexError(f"Cell index {index} out of range (0-{len(self.notebook.cells) - 1})")

        cell = self.notebook.cells[index]
        if not cell.is_code:
            return ExecutionResult(status="ok", outputs=[], error=None)

        timeout = timeout or self._config.default_timeout
        result = await self._executor.execute(cell.source, timeout=timeout)

        # Update cell state
        cell.outputs = result.outputs
        cell.execution_count = result.execution_count

        if result.status == "ok":
            self._staleness.on_execute_cell(self.notebook, index)
        else:
            # On error, cell stays dirty/stale but gets error output
            cell.last_executed_source = cell.source

        self._bump()
        return result

    async def run_from(self, index: int) -> list[ExecutionResult]:
        """Run cell at index and all code cells after it."""
        results: list[ExecutionResult] = []

        code_indices = [index] + self.notebook.downstream_code_indices(index)
        code_indices = [i for i in code_indices if i < len(self.notebook.cells) and self.notebook.cells[i].is_code]

        for i in code_indices:
            result = await self.run_cell(i)
            results.append(result)
            if result.status == "error":
                break  # Stop on first error

        return results

    async def run_stale(self) -> list[ExecutionResult]:
        """Run all DIRTY and STALE code cells in order."""
        results: list[ExecutionResult] = []

        for i, cell in enumerate(self.notebook.cells):
            if cell.is_code and cell.state in (CellState.DIRTY, CellState.STALE):
                result = await self.run_cell(i)
                results.append(result)
                if result.status == "error":
                    break

        return results

    # --- Savepoints ---

    def create_savepoint(self, label: str = "") -> str:
        """Create a savepoint of the current notebook state."""
        sp_id = str(uuid.uuid4())[:8]
        snapshot_path = self._savepoint_dir / f"{sp_id}.ipynb"
        NotebookStore.save(copy.deepcopy(self.notebook), snapshot_path)
        savepoint = Savepoint(
            id=sp_id,
            label=label or f"savepoint-{sp_id}",
            timestamp=time.time(),
            snapshot_path=str(snapshot_path),
            cell_count=len(self.notebook.cells),
        )
        self._savepoints.append(savepoint)
        self._persist_savepoints()
        logger.info("Created savepoint: %s (%s)", sp_id, label)
        return sp_id

    def restore_savepoint(self, savepoint_id: str) -> None:
        """Restore notebook to a previous savepoint."""
        for sp in self._savepoints:
            if sp.id == savepoint_id:
                snapshot = NotebookStore.load(sp.snapshot_path)
                self.notebook.cells = copy.deepcopy(snapshot.cells)
                self.notebook.metadata = copy.deepcopy(snapshot.metadata)
                self._bump()
                logger.info("Restored savepoint: %s", savepoint_id)
                return
        raise ValueError(f"Savepoint {savepoint_id} not found")

    def list_savepoints(self) -> list[dict]:
        """List available savepoints."""
        return [
            {
                "id": sp.id,
                "label": sp.label,
                "timestamp": sp.timestamp,
                "cell_count": sp.cell_count,
            }
            for sp in self._savepoints
        ]
