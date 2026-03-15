"""AlphaEvolve-style multi-session campaign runner.

A high-level orchestrator that runs multiple sequential LLM sessions on a notebook,
passing cross-session context between them. Built on top of the session layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from notebook_agent.config import AgentConfig
from notebook_agent.context.renderer import ContextRenderer
from notebook_agent.engine.llm_interface import LLMInterface
from notebook_agent.kernel.manager import KernelManager
from notebook_agent.notebook_ops.manager import NotebookManager
from notebook_agent.persistence.checkpoint import CheckpointManager
from notebook_agent.persistence.store import NotebookStore
from notebook_agent.session import SessionResult, SessionRunner
from notebook_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class CampaignRunner:
    """Runs a multi-session campaign: sequential sessions with cross-session context."""

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

    async def run(self, task: str) -> None:
        """Run the campaign: one or more sessions on the notebook."""
        max_sessions = self._config.max_sessions
        rounds_per_session = self._config.rounds_per_session or self._config.max_rounds

        logger.info(
            "Starting campaign: %d session(s), %d rounds/session",
            max_sessions, rounds_per_session,
        )

        summaries: list[str] = []
        round_offset = 0

        for session_num in range(1, max_sessions + 1):
            logger.info("=== Session %d/%d ===", session_num, max_sessions)

            # For sessions after the first, restart kernel and reload notebook
            if session_num > 1:
                NotebookStore.save(self._nb.notebook, self._config.notebook_path)
                await self._kernel.restart()
                self._nb.notebook = NotebookStore.load(self._config.notebook_path)
                logger.info("Restarted kernel and reloaded notebook for session %d", session_num)

            task_prompt = self._build_task_prompt(task, session_num, summaries)

            runner = SessionRunner(
                llm=self._llm,
                notebook_manager=self._nb,
                context_renderer=self._renderer,
                tool_registry=self._tools,
                checkpoint_manager=self._checkpoints,
                kernel=self._kernel,
                config=self._config,
            )

            result = await runner.run(task_prompt, rounds_per_session, round_offset)
            round_offset += result.rounds_used

            summary = self._summarize_session(session_num, result)
            summaries.append(summary)
            logger.info("Session %d summary: %s", session_num, summary)

            if result.done and session_num < max_sessions:
                logger.info("Agent signaled completion at session %d", session_num)
                break

    def _build_task_prompt(self, task: str, session_num: int, summaries: list[str]) -> str:
        """Build the task prompt for a session, including cross-session context."""
        parts = [f"## Your Task\n{task}"]

        if self._config.system_prompt_extra:
            parts.append(self._config.system_prompt_extra)

        if summaries:
            history = "\n".join(f"- Session {i+1}: {s}" for i, s in enumerate(summaries))
            parts.append(
                f"## Session History\n"
                f"This is session {session_num}. Previous sessions:\n{history}\n\n"
                f"The notebook contains work from previous sessions. "
                f"Code cells are marked STALE because the kernel was restarted. "
                f"Review the notebook and continue making progress."
            )

        return "\n\n".join(parts)

    def _summarize_session(self, session_num: int, result: SessionResult) -> str:
        """Create a one-line summary of a session."""
        status = "completed" if result.done else f"used all {result.rounds_used} rounds"
        tokens = result.total_usage
        return (
            f"{status} | "
            f"{tokens.get('input_tokens', 0)} in / {tokens.get('output_tokens', 0)} out tokens"
        )
