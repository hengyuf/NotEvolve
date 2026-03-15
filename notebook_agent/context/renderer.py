"""Compact notebook rendering for LLM context."""

from __future__ import annotations

from notebook_agent.models.cell import Cell, CellOutput, CellState, CellType
from notebook_agent.models.notebook import Notebook


class ContextRenderer:
    """Renders a Notebook into compact text for the LLM.

    Rendering rules:
    1. All cells shown in order with [index] prefix.
    2. Last N cells (unfold_last_n) shown unfolded with full source + outputs.
    3. Earlier cells folded by default: [i] TYPE [STATE] | first_line (N lines)
    4. Cells explicitly folded/unfolded by model override defaults.
    5. Outputs cropped to max_output_lines, with expand hint.
    6. STALE/DIRTY annotations on cells and outputs.
    7. Hidden cells (folded=True AND explicitly hidden) omitted entirely.
    """

    def __init__(
        self,
        unfold_last_n: int = 3,
        max_output_lines: int = 30,
        max_source_lines_folded: int = 1,
    ):
        self.unfold_last_n = unfold_last_n
        self.max_output_lines = max_output_lines
        self.max_source_lines_folded = max_source_lines_folded

    def render(self, notebook: Notebook, kernel_alive: bool = True) -> str:
        """Render the full notebook for LLM context."""
        if not notebook.cells:
            return "=== NOTEBOOK (0 cells) ===\n[Empty notebook]"

        total = len(notebook.cells)
        header = f"=== NOTEBOOK ({total} cells, kernel: {'alive' if kernel_alive else 'dead'}) ==="

        # Determine which cells are unfolded
        unfold_start = max(0, total - self.unfold_last_n)

        parts = [header, ""]

        folded_section: list[str] = []
        unfolded_section: list[str] = []

        for i, cell in enumerate(notebook.cells):
            # Determine if this cell should be unfolded
            should_unfold = i >= unfold_start

            # Explicit fold/unfold overrides
            if cell.folded:
                should_unfold = False
            elif cell.folded is False:
                should_unfold = True

            if should_unfold:
                unfolded_section.append(self.render_cell_full(cell, i))
            else:
                folded_section.append(self.render_cell_compact(cell, i))

        if folded_section:
            parts.extend(folded_section)
            parts.append("")

        if unfolded_section:
            parts.extend(unfolded_section)

        return "\n".join(parts)

    def render_cell_compact(self, cell: Cell, index: int) -> str:
        """Render a folded cell as a single line."""
        type_str = cell.cell_type.value.upper()
        state_str = self._state_tag(cell)
        exec_str = f" exec:{cell.execution_count}" if cell.execution_count else ""

        first_line = cell.first_line
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."

        line_info = f"  ({cell.line_count} lines)" if cell.line_count > 1 else ""

        # Show brief output summary for code cells with outputs
        output_hint = ""
        if cell.is_code and cell.outputs:
            total_output_lines = sum(o.line_count for o in cell.outputs)
            has_error = any(o.output_type == "error" for o in cell.outputs)
            if has_error:
                output_hint = "  [has error]"
            elif total_output_lines > 0:
                output_hint = f"  [output: {total_output_lines} lines]"

        return f"[{index}] {type_str}{state_str}{exec_str} | {first_line}{line_info}{output_hint}"

    def render_cell_full(self, cell: Cell, index: int) -> str:
        """Render a full cell with source and outputs."""
        type_str = cell.cell_type.value.upper()
        state_str = self._state_tag(cell)
        exec_str = f" exec:{cell.execution_count}" if cell.execution_count else ""

        parts = [f"--- [{index}] {type_str}{state_str}{exec_str} ---"]
        parts.append(cell.source)

        if cell.is_code:
            if cell.state == CellState.UNEXECUTED:
                parts.append("[Not yet executed]")
            elif cell.state == CellState.DIRTY:
                parts.append("[Source changed, not yet re-executed]")
                if cell.outputs:
                    parts.append("--- stale output ---")
                    for out in cell.outputs:
                        parts.append(self._render_output(out, index, stale=True))
            elif cell.outputs:
                stale = cell.state == CellState.STALE
                parts.append("--- output" + (" [STALE]" if stale else "") + " ---")
                for out in cell.outputs:
                    parts.append(self._render_output(out, index, stale=stale))
            else:
                parts.append("[No output]")

        parts.append("")  # Blank line after cell
        return "\n".join(parts)

    def _render_output(self, output: CellOutput, cell_index: int, stale: bool = False) -> str:
        """Render a single output, applying cropping."""
        text = output.full_content
        if not text:
            return ""

        lines = text.splitlines()
        total = len(lines)

        if total <= self.max_output_lines:
            return text

        # Crop: show first and last portions
        head_n = self.max_output_lines // 2
        tail_n = self.max_output_lines - head_n
        hidden = total - head_n - tail_n

        head = "\n".join(lines[:head_n])
        tail = "\n".join(lines[-tail_n:])

        return (
            f"{head}\n"
            f"... [{hidden} more lines. Use expand_output({cell_index}) to see full]\n"
            f"{tail}"
        )

    def _state_tag(self, cell: Cell) -> str:
        """Return a state annotation string."""
        if cell.is_markdown or cell.is_title:
            return ""
        state_map = {
            CellState.CLEAN: " [CLEAN]",
            CellState.DIRTY: " [DIRTY]",
            CellState.STALE: " [STALE]",
            CellState.UNEXECUTED: " [UNEXECUTED]",
        }
        return state_map.get(cell.state, "")

    def expand_cell(self, notebook: Notebook, index: int) -> str:
        """Return full source of a specific cell."""
        if index < 0 or index >= len(notebook.cells):
            return f"Error: cell index {index} out of range (0-{len(notebook.cells) - 1})"
        cell = notebook.cells[index]
        return self.render_cell_full(cell, index)

    def expand_output(
        self, notebook: Notebook, cell_index: int, output_index: int = 0
    ) -> str:
        """Return full uncropped output of a cell."""
        if cell_index < 0 or cell_index >= len(notebook.cells):
            return f"Error: cell index {cell_index} out of range"
        cell = notebook.cells[cell_index]
        if not cell.outputs:
            return f"Cell [{cell_index}] has no outputs."
        if output_index < 0 or output_index >= len(cell.outputs):
            return f"Error: output index {output_index} out of range (0-{len(cell.outputs) - 1})"

        output = cell.outputs[output_index]
        return output.full_content or "[Empty output]"
