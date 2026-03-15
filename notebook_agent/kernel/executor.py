"""Cell execution on a Jupyter kernel with output collection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import partial

import nbformat
from jupyter_core.utils import ensure_async

from notebook_agent.kernel.manager import KernelManager
from notebook_agent.models.cell import CellOutput

logger = logging.getLogger(__name__)

# ANSI escape code pattern
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


@dataclass
class ExecutionResult:
    """Result of executing a code cell."""

    status: str  # "ok" or "error"
    execution_count: int | None = None
    outputs: list[CellOutput] = field(default_factory=list)
    error: str | None = None


def _output_hook(outputs: list[dict], msg: dict) -> None:
    """Callback collecting outputs during execution.

    Adapted from jupyter-server-nbmodel/actions.py:_output_hook.
    """
    msg_type = msg["header"]["msg_type"]
    if msg_type in ("display_data", "stream", "execute_result", "error"):
        output = nbformat.v4.output_from_msg(msg)
        if msg_type == "stream":
            # Merge consecutive stream outputs of same name
            if outputs and outputs[-1].get("output_type") == "stream" and outputs[-1].get("name") == output.get("name"):
                outputs[-1]["text"] += output["text"]
            else:
                outputs.append(output)
        else:
            outputs.append(output)
    elif msg_type == "clear_output":
        outputs.clear()


class CellExecutor:
    """Executes code on a Jupyter kernel and collects structured outputs."""

    def __init__(self, kernel_manager: KernelManager):
        self._km = kernel_manager

    async def execute(self, code: str, timeout: int = 120) -> ExecutionResult:
        """Execute code and return structured result.

        Uses jupyter_client's execute_interactive with output hook
        for streaming output collection.
        """
        if not self._km.is_alive:
            return ExecutionResult(
                status="error",
                error="Kernel is not running. Use restart_kernel to start it.",
            )

        client = self._km.client
        raw_outputs: list[dict] = []

        try:
            reply = await ensure_async(
                client.execute_interactive(
                    code,
                    output_hook=partial(_output_hook, raw_outputs),
                    timeout=timeout,
                )
            )
        except TimeoutError:
            return ExecutionResult(
                status="error",
                error=f"Execution timed out after {timeout} seconds.",
            )
        except Exception as e:
            logger.error("Execution failed: %s", e, exc_info=True)
            return ExecutionResult(
                status="error",
                error=f"Execution failed: {e}",
            )

        reply_content = reply["content"]
        status = reply_content.get("status", "error")

        # Parse raw outputs into CellOutput objects
        parsed_outputs = [CellOutput.from_nbformat(out) for out in raw_outputs]

        error_msg = None
        if status == "error":
            error_parts = []
            for out in parsed_outputs:
                if out.output_type == "error":
                    error_parts.append(out.content)
            error_msg = "\n".join(error_parts) if error_parts else "Unknown error"

        return ExecutionResult(
            status=status,
            execution_count=reply_content.get("execution_count"),
            outputs=parsed_outputs,
            error=error_msg,
        )
