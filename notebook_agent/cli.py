"""CLI entry point for notebook-agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import click
from click.core import ParameterSource

from notebook_agent.config import AgentConfig


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("jupyter_client").setLevel(logging.WARNING)
    logging.getLogger("traitlets").setLevel(logging.WARNING)


def resolve_api_key(provider: str, explicit_api_key: str = "") -> str:
    """Resolve API key from explicit value or provider-appropriate env vars."""
    if explicit_api_key:
        return explicit_api_key

    normalized = provider.lower()
    if normalized == "anthropic":
        candidates = ("ANTHROPIC_API_KEY", "LLM_API_KEY")
    elif normalized in ("openai", "openai_compatible"):
        candidates = ("OPENAI_API_KEY", "LLM_API_KEY")
    else:
        candidates = ("LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")

    for key in candidates:
        value = os.getenv(key, "")
        if value:
            return value
    return ""


async def run_agent(config: AgentConfig) -> None:
    """Set up all components and run the agent loop."""
    from notebook_agent.context.renderer import ContextRenderer
    from notebook_agent.engine.llm_interface import LLMInterface
    from notebook_agent.orchestrators.campaign import CampaignRunner
    from notebook_agent.kernel.executor import CellExecutor
    from notebook_agent.kernel.manager import KernelManager
    from notebook_agent.notebook_ops.manager import NotebookManager
    from notebook_agent.persistence.checkpoint import CheckpointManager
    from notebook_agent.persistence.store import NotebookStore
    from notebook_agent.security.policy import SecurityPolicy
    from notebook_agent.tools.context_tools import FoldCellTool, UnfoldCellTool
    from notebook_agent.tools.evaluator_tools import load_evaluator_tools
    from notebook_agent.tools.file_tools import (
        GlobFilesTool,
        GrepFilesTool,
        ListTreeTool,
        ReadFileTool,
    )
    from notebook_agent.tools.notebook_tools import (
        AddCellTool,
        CreateSavepointTool,
        DeleteCellTool,
        EditCellTool,
        ExpandOutputTool,
        ListSavepointsTool,
        ReadCellTool,
        ReadNotebookTool,
        RestartKernelTool,
        RestoreSavepointTool,
        RunCellTool,
        RunFromCellTool,
        RunStaleCellsTool,
    )
    from notebook_agent.tools.registry import ToolRegistry

    logger = logging.getLogger(__name__)

    # Normalize key filesystem paths up front.
    working_dir = Path(config.working_dir).resolve()
    config.working_dir = str(working_dir)

    notebook_path = Path(config.notebook_path)
    if not notebook_path.is_absolute():
        notebook_path = working_dir / notebook_path
    config.notebook_path = str(notebook_path.resolve())

    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = working_dir / checkpoint_dir
    config.checkpoint_dir = str(checkpoint_dir.resolve())

    evaluator_paths = list(config.evaluator_paths)
    if config.evaluator_module and config.evaluator_module.endswith(".py"):
        eval_module_path = Path(config.evaluator_module)
        if not eval_module_path.is_absolute():
            eval_module_path = working_dir / eval_module_path
        evaluator_paths.append(str(eval_module_path.resolve()))

    # Initialize security
    security = SecurityPolicy(
        forbidden_code_patterns=config.forbidden_code_patterns,
        evaluator_paths=evaluator_paths,
        working_dir=config.working_dir,
    )

    # Initialize kernel
    kernel = KernelManager(kernel_name=config.kernel_name)
    await kernel.start()

    try:
        executor = CellExecutor(kernel)

        # Load or create notebook
        nb_path = Path(config.notebook_path)
        if nb_path.exists():
            notebook = NotebookStore.load(nb_path)
            logger.info("Loaded existing notebook: %s (%d cells)", nb_path, len(notebook))
        else:
            notebook = NotebookStore.create_empty(nb_path)
            logger.info("Created new notebook: %s", nb_path)

        # Initialize managers
        nb_manager = NotebookManager(notebook, executor, config)
        renderer = ContextRenderer(
            unfold_last_n=config.unfold_last_n,
            max_output_lines=config.max_output_lines,
        )
        checkpoint_mgr = CheckpointManager(config.checkpoint_dir, config.notebook_path)

        # Register tools
        registry = ToolRegistry(security=security)
        registry.register_all([
            # Notebook reading
            ReadNotebookTool(nb_manager, renderer, kernel),
            ReadCellTool(nb_manager, renderer),
            ExpandOutputTool(nb_manager, renderer),
            # Notebook writing
            AddCellTool(nb_manager),
            EditCellTool(nb_manager),
            DeleteCellTool(nb_manager),
            # Execution
            RunCellTool(nb_manager, security),
            RunFromCellTool(nb_manager, security),
            RunStaleCellsTool(nb_manager, security),
            # Context management
            FoldCellTool(nb_manager),
            UnfoldCellTool(nb_manager),
            # File operations
            GlobFilesTool(config.working_dir, security),
            GrepFilesTool(config.working_dir, security),
            ReadFileTool(config.working_dir, security),
            ListTreeTool(config.working_dir, security),
            # Kernel management
            RestartKernelTool(kernel, nb_manager),
            CreateSavepointTool(nb_manager),
            ListSavepointsTool(nb_manager),
            RestoreSavepointTool(nb_manager),
        ])

        # Register evaluator tools if configured
        if config.evaluator_module:
            module_spec = config.evaluator_module
            if module_spec.endswith(".py"):
                module_path = Path(module_spec)
                if not module_path.is_absolute():
                    module_path = Path(config.working_dir) / module_path
                module_spec = str(module_path.resolve())

            eval_tools = load_evaluator_tools(
                {"module": module_spec, "tools": config.evaluator_tools}
            )
            if eval_tools:
                registry.register_all(eval_tools)
                logger.info(
                    "Registered %d evaluator tool(s) from %s",
                    len(eval_tools),
                    module_spec,
                )
            else:
                logger.warning("No evaluator tools found in %s", module_spec)

        # Initialize LLM
        if config.llm_provider == "human":
            from notebook_agent.engine.human_llm import HumanLLM
            llm = HumanLLM()
        else:
            llm = LLMInterface(
                provider=config.llm_provider,
                model=config.llm_model,
                api_key=config.llm_api_key,
                max_tokens=config.llm_max_tokens,
                thinking_budget=config.llm_thinking_budget,
                base_url=config.llm_base_url,
                extra_headers=config.llm_extra_headers,
            )

        # Initialize campaign runner
        campaign = CampaignRunner(
            llm=llm,
            notebook_manager=nb_manager,
            context_renderer=renderer,
            tool_registry=registry,
            checkpoint_manager=checkpoint_mgr,
            kernel=kernel,
            config=config,
        )

        # Run
        await campaign.run(task=config.task)

    finally:
        await kernel.shutdown()


@click.command()
@click.option("--task", "-t", default="", help="Task description for the agent.")
@click.option("--notebook", "-n", default="notebook.ipynb", help="Path to notebook file.")
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="LLM model name.")
@click.option("--provider", "-p", default="anthropic", help="LLM provider (anthropic, openai, openai_compatible, human).")
@click.option("--api-key", default="", help="API key for selected provider.")
@click.option("--base-url", default="", help="Optional API base URL for OpenAI-compatible providers.")
@click.option("--max-rounds", default=50, help="Maximum number of rounds.")
@click.option("--max-sessions", default=1, help="Number of sessions (1 = single session).")
@click.option("--rounds-per-session", default=0, help="Rounds per session (0 = use max-rounds).")
@click.option("--max-tokens", default=16384, help="Max tokens per LLM response.")
@click.option("--thinking-budget", default=10000, help="Extended thinking budget tokens.")
@click.option("--working-dir", "-w", default=".", help="Working directory.")
@click.option("--checkpoint-dir", default=".checkpoints", help="Checkpoint directory.")
@click.option("--evaluator-path", multiple=True, help="Protected evaluator paths.")
@click.option("--evaluator-module", default="", help="Python module/file with evaluator functions.")
@click.option("--kernel", default="python3", help="Jupyter kernel name.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option("--config-file", "-c", default=None, help="Path to config file (JSON).")
@click.pass_context
def main(
    ctx: click.Context,
    task: str,
    notebook: str,
    model: str,
    provider: str,
    api_key: str,
    base_url: str,
    max_rounds: int,
    max_sessions: int,
    rounds_per_session: int,
    max_tokens: int,
    thinking_budget: int,
    working_dir: str,
    checkpoint_dir: str,
    evaluator_path: tuple[str, ...],
    evaluator_module: str,
    kernel: str,
    verbose: bool,
    config_file: str | None,
) -> None:
    """Run the notebook-agent scaffolding system."""
    setup_logging(verbose)

    config_kwargs: dict = {}
    if config_file:
        with open(config_file, encoding="utf-8") as f:
            file_config = json.load(f)
        if isinstance(file_config, dict):
            config_kwargs.update(file_config)

    cli_to_config = {
        "task": ("task", task),
        "notebook": ("notebook_path", notebook),
        "model": ("llm_model", model),
        "provider": ("llm_provider", provider),
        "api_key": ("llm_api_key", api_key),
        "base_url": ("llm_base_url", base_url),
        "max_rounds": ("max_rounds", max_rounds),
        "max_sessions": ("max_sessions", max_sessions),
        "rounds_per_session": ("rounds_per_session", rounds_per_session),
        "max_tokens": ("llm_max_tokens", max_tokens),
        "thinking_budget": ("llm_thinking_budget", thinking_budget),
        "working_dir": ("working_dir", working_dir),
        "checkpoint_dir": ("checkpoint_dir", checkpoint_dir),
        "evaluator_path": ("evaluator_paths", list(evaluator_path)),
        "evaluator_module": ("evaluator_module", evaluator_module),
        "kernel": ("kernel_name", kernel),
    }

    for cli_param, (cfg_key, value) in cli_to_config.items():
        source = ctx.get_parameter_source(cli_param)
        if source not in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP):
            config_kwargs[cfg_key] = value

    if config_file:
        config_base = Path(config_file).resolve().parent
        if config_kwargs.get("working_dir"):
            wd = Path(config_kwargs["working_dir"])
            if not wd.is_absolute():
                config_kwargs["working_dir"] = str((config_base / wd).resolve())
        else:
            config_kwargs["working_dir"] = str(config_base)

        wd = Path(config_kwargs["working_dir"])
        for path_key in ("notebook_path", "checkpoint_dir"):
            if config_kwargs.get(path_key):
                p = Path(config_kwargs[path_key])
                if not p.is_absolute():
                    config_kwargs[path_key] = str((wd / p).resolve())

        if config_kwargs.get("evaluator_module"):
            module_path = config_kwargs["evaluator_module"]
            if isinstance(module_path, str) and module_path.endswith(".py"):
                p = Path(module_path)
                if not p.is_absolute():
                    config_kwargs["evaluator_module"] = str((wd / p).resolve())

    config_kwargs["task"] = config_kwargs.get("task", task)
    if not config_kwargs.get("task"):
        raise click.UsageError("Task is required (provide --task or set 'task' in --config-file).")

    provider_name = str(config_kwargs.get("llm_provider", provider))
    if provider_name != "human":
        explicit_key = str(config_kwargs.get("llm_api_key", "") or "")
        resolved_key = resolve_api_key(provider_name, explicit_key)
        if resolved_key:
            config_kwargs["llm_api_key"] = resolved_key

    config = AgentConfig(**config_kwargs)

    try:
        asyncio.run(run_agent(config))
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
