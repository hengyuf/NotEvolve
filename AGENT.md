# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Training-free scaffolding that lets an agentic LLM work iteratively in a notebook-style environment with persistent execution state. The LLM gets a live Jupyter kernel, a compact notebook view, and coding-agent-style file tools. It writes code cells, runs them, inspects outputs, and refines its approach over many rounds — without rerunning the entire notebook each time.

LLM backend supports Anthropic and OpenAI-compatible APIs (set `--provider openai_compatible` and `--base-url` for custom endpoints).

The `references/` directory contains external reference projects (jupyter-mcp-server, jupyter-mcp-tools, etc.) — these are not part of this project's source.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_context.py -v

# Run a single test
pytest tests/test_context.py::test_render_folded_cells -v

# Run the agent
notebook-agent --task "..." --notebook work.ipynb --max-rounds 20

# Run with a config file
notebook-agent --config-file examples/beat_the_average_game/config.json --task "..."

# Debug logging
notebook-agent --task "..." --verbose
```

## Architecture

Two-layer design:

1. **Low-level scaffolding** (the session layer, `session.py` + everything it imports): The general-purpose framework for running one LLM session on a notebook. This includes the round loop (`SessionRunner`), notebook rendering (`ContextRenderer`), tool dispatch (`ToolRegistry`), LLM interaction (`LLMInterface`), kernel management, security, and persistence. It takes an opaque task prompt string and runs rounds — it has no concept of campaigns, cross-session state, or problem-specific logic. All modules under `notebook_agent/` except `orchestrators/` and `cli.py` belong to this layer.
2. **High-level orchestrators** (`orchestrators/`): Consumers of the session layer that implement strategies for using notebook sessions to accomplish a goal. They own task-specific prompt construction, cross-session steering, and when/how to restart kernels or reload notebooks. `CampaignRunner` (`orchestrators/campaign.py`) is the first example — an AlphaEvolve-style multi-session loop. For single-session runs (`max_sessions=1`, the default), it degenerates to one session with `max_rounds` rounds. New orchestrators (e.g., population-based search, tournament selection) should be added here as additional consumers of `SessionRunner`.

### Data flow per round

```
SessionRunner._run_round()
  → ContextRenderer.render(notebook)        # compact plain-text notebook view
  → LLMInterface.chat(system, messages)      # call LLM with notebook context + tools
  → tool call loop:
      ToolRegistry.dispatch(name, args)
        → SecurityPolicy.check_tool_call()   # path sandbox + evaluator isolation
        → tool.execute(**args)               # may call CellExecutor, NotebookManager, etc.
      (repeat until LLM returns text-only or hits max_tool_calls_per_round)
  → CheckpointManager.save()                 # snapshot round state
```

### Key subsystems

- **NotebookManager** (`notebook_ops/manager.py`): Central mutation point for all cell CRUD and execution. Maintains staleness invariants via `StalenessTracker`.
- **StalenessTracker** (`notebook_ops/staleness.py`): Linear downstream propagation — editing a code cell marks it DIRTY and all downstream code cells STALE. No dependency graph; position-based only.
- **ContextRenderer** (`context/renderer.py`): Renders the notebook as structured plain text. Folding (compact one-liners for old cells) + output cropping with `expand_output()` hints.
- **SecurityPolicy** (`security/policy.py`): Workspace sandbox (file paths must stay inside `working_dir`), evaluator file isolation, forbidden code pattern scanning (pip install, subprocess, os.system).
- **ToolRegistry** (`tools/registry.py`): Maps tool names to handlers, gates every call through SecurityPolicy.
- **LLMInterface** (`engine/llm_interface.py`): Adapter for Anthropic Messages API and OpenAI Chat Completions API. Internally stores messages in Anthropic format; converts to OpenAI format on the fly.
- **Evaluator tools** (`tools/evaluator_tools.py`): Auto-discovered from `--evaluator-module`. Public functions starting with `evaluate` or `check` become LLM-callable tools. Evaluator source is blocked from the LLM by SecurityPolicy.

### Tool definition pattern

All tools inherit `BaseTool` (`tools/base.py`), implementing `name`, `description`, `parameters` (JSON Schema), and `async execute(**kwargs) -> ToolResult`. Schemas use Anthropic's `input_schema` key format; LLMInterface converts to OpenAI's `parameters` format when needed.

### Notebook persistence

`NotebookStore` (`persistence/store.py`) reads/writes standard `.ipynb` (nbformat v4). Custom agent metadata (cell state, folded, stale_reason) is stored under each cell's `metadata.notebook_agent` key. On load, code cells with existing outputs are marked STALE since kernel state isn't persisted.

## Testing

- Tests use `pytest` with `pytest-asyncio` (configured `asyncio_mode = "auto"` in pyproject.toml).
- Shared fixtures in `tests/conftest.py`: `sample_notebook` (5-cell notebook with mixed types/states), `config` (AgentConfig with temp paths), `tmp_dir`.
- Kernel tests (`test_kernel.py`) start a real ipykernel subprocess — they are slower and require a working Jupyter kernel.
- Most other tests are pure unit tests with no kernel dependency.
