"""Session layer: a self-contained framework for running one LLM session on a notebook."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from notebook_agent.config import AgentConfig
from notebook_agent.context.renderer import ContextRenderer
from notebook_agent.engine.llm_interface import LLMInterface
from notebook_agent.kernel.manager import KernelManager
from notebook_agent.notebook_ops.manager import NotebookManager
from notebook_agent.persistence.checkpoint import CheckpointManager
from notebook_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SCAFFOLD_PROMPT = """\
You are an expert research agent working in a notebook environment.

## Environment
You have a persistent Jupyter notebook with a live Python kernel. Objects and variables persist across cell executions within a round and across rounds. You do NOT need to re-import or re-define things that are already in the kernel's memory.

## Available Tools
You have tools for:
- **Notebook reading**: read_notebook, read_cell, expand_output
- **Notebook writing**: add_cell (code/markdown/title), edit_cell, delete_cell
- **Execution**: run_cell, run_from_cell, run_stale_cells
- **Context management**: fold_cell, unfold_cell
- **File operations**: glob_files, grep_files, read_file, list_tree
- **Kernel management**: restart_kernel, create_savepoint, list_savepoints, restore_savepoint
{evaluator_tools_desc}

## Notebook Conventions
- Cell states: CLEAN (executed, up-to-date), DIRTY (edited, needs re-run), STALE (upstream changed), UNEXECUTED (never run)
- When you edit a code cell, it becomes DIRTY and downstream code cells become STALE
- Always run cells after editing to update the kernel state
- When deleting cells for cleanup, insert a summary markdown cell first to preserve context
- Long outputs are cropped by default; use expand_output() to see the full output
- Earlier cells are folded (compact view); use read_cell() or unfold_cell() to see details

## Strategy
- Work iteratively: write code, run it, inspect results, refine
- Use markdown cells for notes and observations
- Create savepoints before major changes
- Be systematic in your search/optimization approach
"""


@dataclass
class SessionResult:
    """Result of a single session run."""

    rounds_used: int = 0
    total_usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    done: bool = False


class SessionRunner:
    """Runs a single LLM session: a sequence of rounds where the LLM interacts with a notebook via tool calls."""

    def __init__(
        self,
        llm: LLMInterface,
        notebook_manager: NotebookManager,
        context_renderer: ContextRenderer,
        tool_registry: ToolRegistry,
        checkpoint_manager: CheckpointManager,
        kernel: KernelManager,
        config: AgentConfig,
    ):
        self._llm = llm
        self._nb = notebook_manager
        self._renderer = context_renderer
        self._tools = tool_registry
        self._checkpoints = checkpoint_manager
        self._kernel = kernel
        self._config = config
        self._total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    async def run(self, task_prompt: str, max_rounds: int, round_offset: int = 0) -> SessionResult:
        """Run a session: a sequence of rounds on the notebook.

        Args:
            task_prompt: Opaque string from the caller — could be a simple task,
                a rich multi-page brief, anything.
            max_rounds: Maximum number of rounds for this session.
            round_offset: For globally sequential checkpoint numbering
                (caller's responsibility).

        Returns:
            SessionResult with stats about the session.
        """
        self._total_usage = {"input_tokens": 0, "output_tokens": 0}

        logger.info("Starting session: %s (max %d rounds)", task_prompt[:100], max_rounds)

        done = False
        rounds_used = 0
        for round_num in range(1, max_rounds + 1):
            global_round = round_offset + round_num
            logger.info("=== Round %d/%d (global %d) ===", round_num, max_rounds, global_round)
            round_start = time.time()

            done = await self._run_round(global_round, task_prompt)
            rounds_used = round_num

            elapsed = time.time() - round_start
            logger.info(
                "Round %d completed in %.1fs (tokens: in=%d, out=%d)",
                round_num, elapsed,
                self._total_usage.get("input_tokens", 0),
                self._total_usage.get("output_tokens", 0),
            )

            if done:
                logger.info("Agent signaled completion at round %d", round_num)
                break
        else:
            logger.warning("Reached max rounds (%d) without completion", max_rounds)

        return SessionResult(
            rounds_used=rounds_used,
            total_usage=dict(self._total_usage),
            done=done,
        )

    async def _run_round(self, round_number: int, task_prompt: str) -> bool:
        """Execute one round of interaction. Returns True if agent signals done."""
        # Build system prompt
        system = self._build_system_prompt(task_prompt)

        # Render notebook context
        notebook_context = self._renderer.render(self._nb.notebook, self._kernel.is_alive)

        # Build initial messages for this round
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"## Current Notebook State\n\n{notebook_context}\n\n"
                    f"## Round {round_number}\n"
                    f"Continue working on the task. Use the available tools to make progress."
                ),
            }
        ]

        tool_calls_this_round = 0
        all_tool_calls: list[dict] = []

        while True:
            # Call LLM
            response = await self._llm.chat(
                messages=messages,
                tools=self._tools.get_schemas() or None,
                temperature=self._config.llm_temperature,
                system=system,
            )

            # Track usage
            for k, v in response.usage.items():
                self._total_usage[k] = self._total_usage.get(k, 0) + v

            # If no tool calls, round ends
            if not response.tool_calls:
                if response.content:
                    logger.info("Agent response: %s", response.content[:200])
                break

            # Process tool calls
            # Add assistant message with tool use
            assistant_content: list[dict] = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })

            messages.append({"role": "assistant", "content": assistant_content})

            # Dispatch tool calls and collect results
            tool_result_content: list[dict] = []
            for tc in response.tool_calls:
                tool_calls_this_round += 1
                logger.info("Tool call: %s(%s)", tc.name, json.dumps(tc.arguments)[:200])

                result = await self._tools.dispatch(tc.name, tc.arguments)
                logger.info("Tool result: %s", result.content[:200] if result.content else "(empty)")

                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result.content,
                    "is_error": result.is_error,
                })

                all_tool_calls.append({
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": result.content[:500],
                    "is_error": result.is_error,
                })

            messages.append({"role": "user", "content": tool_result_content})

            # Check tool call limit
            if tool_calls_this_round >= self._config.max_tool_calls_per_round:
                logger.warning("Reached tool call limit (%d) for round %d",
                             self._config.max_tool_calls_per_round, round_number)
                # Send a message telling the LLM to wrap up
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have reached the tool call limit ({self._config.max_tool_calls_per_round}) "
                        f"for this round. Please provide a summary of your progress and what to do next."
                    ),
                })
                # Get final response without tools
                final_response = await self._llm.chat(
                    messages=messages,
                    temperature=self._config.llm_temperature,
                    system=system,
                )
                if final_response.content:
                    logger.info("Round summary: %s", final_response.content[:200])
                break

        # Create checkpoint
        self._checkpoints.save(
            round_number=round_number,
            notebook=self._nb.notebook,
            messages=messages,
            tool_calls=all_tool_calls,
        )

        # Check if agent is done (stop_reason == "end_turn" with no tool calls)
        return response.stop_reason == "end_turn" and not response.tool_calls

    def _build_system_prompt(self, task_prompt: str) -> str:
        """Build the system prompt: task_prompt + scaffold."""
        # Build evaluator tools description
        evaluator_desc = ""
        evaluator_tools = [
            s for s in self._tools.get_schemas()
            if s["name"].startswith("evaluate") or s["name"].startswith("check")
        ]
        if evaluator_tools:
            lines = ["- **Evaluator**: "]
            for t in evaluator_tools:
                lines.append(f"  - {t['name']}: {t.get('description', '')}")
            evaluator_desc = "\n".join(lines)

        scaffold = SCAFFOLD_PROMPT.format(evaluator_tools_desc=evaluator_desc)
        return task_prompt + "\n\n" + scaffold
