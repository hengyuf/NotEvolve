"""Per-round checkpoint management."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from notebook_agent.models.notebook import Notebook
from notebook_agent.persistence.store import NotebookStore


@dataclass
class Checkpoint:
    """A snapshot of the agent state at the end of a round."""

    round_number: int
    timestamp: float
    notebook_path: str
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


class CheckpointManager:
    """Manages per-round checkpoints."""

    def __init__(self, checkpoint_dir: str, notebook_path: str):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._notebook_path = notebook_path

    def save(
        self,
        round_number: int,
        notebook: Notebook,
        messages: list[dict],
        tool_calls: list[dict],
    ) -> str:
        """Save checkpoint for a round.

        Saves both the notebook state and the conversation history.
        """
        # Save notebook snapshot
        nb_snapshot_path = self._dir / f"round_{round_number:04d}_notebook.ipynb"
        NotebookStore.save(notebook, nb_snapshot_path)

        # Save conversation state
        checkpoint = Checkpoint(
            round_number=round_number,
            timestamp=time.time(),
            notebook_path=str(nb_snapshot_path),
            messages=self._sanitize_messages(messages),
            tool_calls=tool_calls,
        )

        conv_path = self._dir / f"round_{round_number:04d}_state.json"
        with open(conv_path, "w", encoding="utf-8") as f:
            json.dump(asdict(checkpoint), f, indent=2, default=str)

        return str(conv_path)

    def load(self, round_number: int) -> Checkpoint | None:
        """Load a checkpoint by round number."""
        conv_path = self._dir / f"round_{round_number:04d}_state.json"
        if not conv_path.exists():
            return None

        with open(conv_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Checkpoint(**data)

    def latest(self) -> Checkpoint | None:
        """Load the most recent checkpoint."""
        files = sorted(self._dir.glob("round_*_state.json"))
        if not files:
            return None

        with open(files[-1], "r", encoding="utf-8") as f:
            data = json.load(f)

        return Checkpoint(**data)

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints."""
        results = []
        for f in sorted(self._dir.glob("round_*_state.json")):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                results.append({
                    "round": data["round_number"],
                    "timestamp": data["timestamp"],
                    "file": str(f),
                })
        return results

    @staticmethod
    def _sanitize_messages(messages: list[dict]) -> list[dict]:
        """Sanitize messages for JSON serialization.

        Truncates very long content to keep checkpoint files manageable.
        """
        sanitized = []
        for msg in messages:
            m = dict(msg)
            if isinstance(m.get("content"), str) and len(m["content"]) > 50000:
                m["content"] = m["content"][:50000] + "\n... [truncated]"
            sanitized.append(m)
        return sanitized
