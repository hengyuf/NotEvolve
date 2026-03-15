"""Notebook data model."""

from __future__ import annotations

from dataclasses import dataclass, field

from notebook_agent.models.cell import Cell, CellType


@dataclass
class Notebook:
    """A notebook containing an ordered list of cells."""

    cells: list[Cell] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    version: int = 0  # Incremented on every mutation

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, idx: int) -> Cell:
        return self.cells[idx]

    def bump_version(self) -> int:
        """Increment and return the new version number."""
        self.version += 1
        return self.version

    def code_cell_indices(self) -> list[int]:
        """Return indices of all code cells, in order."""
        return [i for i, c in enumerate(self.cells) if c.cell_type == CellType.CODE]

    def downstream_code_indices(self, from_idx: int) -> list[int]:
        """Return indices of code cells strictly after from_idx."""
        return [i for i in self.code_cell_indices() if i > from_idx]
