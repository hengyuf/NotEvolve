# NotEvolve

**NotEvolve** is a training-free framework for long-horizon LLM agents that use a
Jupyter notebook as their working state. The notebook is not just an interface:
it is the agent's memory, executable environment, and evolving problem-solving
trajectory.

This repository was developed as a course project for **EE290/194: Scalable AI**.

The current package is installed as `notebook-agent` and exposes a CLI for
running an agent over a live `.ipynb` file. The agent can write code cells, run
them in a persistent kernel, inspect outputs, summarize or fold old work, call
trusted evaluator functions, and checkpoint progress across many rounds.

## Why Notebooks?

Notebook state is a useful world model for agentic problem solving because it
combines three roles in one artifact:

- **State to evolve:** code cells, markdown plans, outputs, summaries, and
  checkpoints form a concrete trajectory that can be refined over time.
- **Multimodal memory:** the notebook can preserve text, tables, plots, errors,
  and experiment results without flattening everything into a chat transcript.
- **Executable environment:** a live Jupyter kernel keeps variables, cached
  computations, loaded data, and solver objects available across cells.

This makes NotEvolve especially natural for scientific computing, mathematical
optimization, data analysis, and machine-learning engineering workflows where
progress depends on iterative experimentation.

## Features

- Persistent Jupyter kernel backed by a real `.ipynb` notebook.
- Compact notebook renderer that turns verbose notebook JSON into LLM-readable
  text.
- Context-management tools for folding, unfolding, reading cells, and expanding
  clipped outputs.
- Notebook-editing and execution tools: add, edit, delete, run one cell, run
  from a cell, and run dirty or stale cells.
- File tools scoped to the working directory.
- Trusted evaluator tools auto-discovered from Python functions named
  `evaluate*` or `check*`.
- Evaluator-source protection so the LLM can call scoring functions without
  reading their implementation.
- Per-round checkpoints plus explicit savepoints for major changes.
- Provider support for Anthropic, OpenAI, OpenAI-compatible APIs, and an
  interactive `human` backend for debugging.
- A campaign runner that supports one or more sequential sessions over the same
  notebook.

## Installation

NotEvolve requires Python 3.11 or newer.

```bash
cd /path/to/NotEvolve
pip install -e ".[dev]"
```

Set an API key for the provider you want to use:

```bash
# Anthropic
export ANTHROPIC_API_KEY=...

# OpenAI or OpenAI-compatible providers
export OPENAI_API_KEY=...

# Generic fallback used by both providers
export LLM_API_KEY=...
```

For Gemini through the OpenAI-compatible endpoint, either pass the key with
`--api-key "$GEMINI_API_KEY"` or export it as `OPENAI_API_KEY`.

## Quick Start

Run a notebook agent on a task:

```bash
notebook-agent \
  --task "Find a discrete probability distribution maximizing P[X1+X2+X3 < 2X4]." \
  --notebook work.ipynb \
  --max-rounds 20
```

Equivalent module invocation:

```bash
python -m notebook_agent --task "Explore the task and build a solution." --notebook work.ipynb
```

Run with an OpenAI-compatible model:

```bash
notebook-agent \
  --task "Optimize the objective and report the best score." \
  --provider openai_compatible \
  --model gemini-2.5-pro \
  --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" \
  --api-key "$GEMINI_API_KEY" \
  --notebook work.ipynb \
  --max-rounds 20
```

Run the included worked example:

```bash
export GEMINI_API_KEY=...
bash examples/beat_the_average_game/run.sh --max-rounds 10
```

## How It Works

At a high level, NotEvolve wires together a notebook, a live kernel, an LLM, and
a tool pool:

```text
CLI / config
  -> AgentConfig
  -> KernelManager + NotebookStore + NotebookManager
  -> ContextRenderer + ToolRegistry + LLMInterface
  -> CampaignRunner
      -> SessionRunner
          -> repeated notebook-agent rounds
```

Each round follows the same inner loop:

1. **Render context:** `ContextRenderer` converts the current notebook into a
   compact text view. Recent or explicitly unfolded cells show source and
   outputs; older folded cells are represented by short summaries.
2. **Plan and act:** the LLM receives the task, scaffold prompt, notebook
   snapshot, and JSON-schema tool definitions.
3. **Use tools:** the LLM calls tools to add, edit, run, inspect, fold, unfold,
   or delete notebook cells; it can also read workspace files and call evaluator
   functions.
4. **Preserve state:** executed cells update both the `.ipynb` file and the live
   kernel state. Long outputs are clipped in context but can be expanded.
5. **Checkpoint:** the runner saves the notebook and a per-round checkpoint with
   messages and tool-call logs.

The built-in `CampaignRunner` can run multiple sequential sessions. Between
sessions, it saves the notebook, restarts the kernel, reloads the notebook, and
adds a short session history to the next prompt. More advanced population or
branch-based evolution strategies can be implemented by adding new orchestrators
on top of the same session layer.

## Context Management

Raw Jupyter notebooks are verbose JSON documents with metadata, MIME bundles,
execution counters, and potentially long outputs. NotEvolve keeps the prompt
compact by rendering notebooks into structured text:

```text
=== NOTEBOOK (5 cells, kernel: alive) ===
[0] MARKDOWN | # Plan
[1] CODE [CLEAN] exec:1 | import numpy as np  [output: 3 lines]

--- [3] CODE [DIRTY] ---
def improve_solution(...):
    ...
[Source changed, not yet re-executed]

--- [4] CODE [CLEAN] exec:4 ---
evaluate_solution(candidate)
--- output ---
score = 0.8123
```

The renderer and tools support:

- folding old cells into one-line descriptions;
- explicitly unfolding important cells;
- clipping long outputs with pointers to `expand_output`;
- marking code cells as `CLEAN`, `DIRTY`, `STALE`, or `UNEXECUTED`;
- preserving short markdown summaries before deleting or folding old attempts.

## Evaluator Tools

If `--evaluator-module` is set, NotEvolve loads public Python functions whose
names start with `evaluate` or `check` and exposes them as LLM-callable tools.

Example:

```python
def evaluate_distribution(support: list[float], probs: list[float]) -> str:
    """Return the exact score for a candidate distribution."""
    ...

def check_score() -> str:
    """Return the best score seen so far."""
    ...
```

Run with evaluator protection:

```bash
notebook-agent \
  --task "Search for a high-scoring distribution." \
  --evaluator-module evaluator_adapter.py \
  --evaluator-path evaluator.py \
  --evaluator-path evaluator_adapter.py
```

The LLM can call the evaluator tools and see their results, but file tools are
blocked from reading protected evaluator paths.

## Configuration

All CLI options can be placed in a JSON config file. Explicit CLI arguments
override config values.

```json
{
  "notebook_path": "notebook.ipynb",
  "working_dir": ".",
  "kernel_name": "python3",
  "llm_provider": "openai_compatible",
  "llm_model": "gemini-2.5-pro",
  "llm_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
  "llm_thinking_budget": 0,
  "max_rounds": 30,
  "max_tool_calls_per_round": 30,
  "unfold_last_n": 4,
  "max_output_lines": 40,
  "evaluator_module": "evaluator_adapter.py",
  "evaluator_paths": ["evaluator.py", "evaluator_adapter.py"],
  "checkpoint_dir": ".checkpoints"
}
```

Then run:

```bash
notebook-agent --config-file config.json --task "Your task description"
```

## CLI Reference

```text
notebook-agent [OPTIONS]

Core options:
  -t, --task TEXT               Task description for the agent
  -n, --notebook PATH           Notebook file path
  -m, --model TEXT              LLM model name
  -p, --provider TEXT           anthropic, openai, openai_compatible, or human
  --api-key TEXT                API key for selected provider
  --base-url TEXT               Base URL for OpenAI-compatible providers
  --max-rounds INTEGER          Maximum number of rounds
  --max-sessions INTEGER        Number of sequential sessions
  --rounds-per-session INTEGER  Rounds per session
  -w, --working-dir PATH        Workspace root for file tools
  --checkpoint-dir PATH         Directory for checkpoints
  --evaluator-module TEXT       Python module/file with evaluator functions
  --evaluator-path PATH         Protected evaluator path, repeatable
  --kernel TEXT                 Jupyter kernel name
  -c, --config-file PATH        JSON config file
  -v, --verbose                 Debug logging
```

Use `notebook-agent --help` for the full list of options.

## Project Structure

```text
notebook_agent/
  cli.py                  CLI entry point and component wiring
  config.py               Pydantic configuration model
  session.py              Core single-session round loop and scaffold prompt
  orchestrators/          Higher-level session consumers
    campaign.py           Sequential multi-session campaign runner
  context/
    renderer.py           Notebook-to-text rendering, folding, output clipping
  kernel/
    manager.py            Jupyter kernel lifecycle
    executor.py           Cell execution and output collection
  models/
    cell.py               Cell, output, type, and state models
    notebook.py           In-memory notebook representation
  notebook_ops/
    manager.py            Notebook mutation, execution, savepoints
    staleness.py          Dirty/stale propagation after edits
  tools/
    notebook_tools.py     Notebook read/write/run/savepoint tools
    context_tools.py      Fold/unfold tools
    file_tools.py         Workspace-scoped file tools
    evaluator_tools.py    Evaluator auto-discovery
    registry.py           Tool dispatch and security gate
  security/
    policy.py             Path protection and code-pattern checks
  persistence/
    store.py              .ipynb serialization
    checkpoint.py         Per-round checkpointing
```

## Development

```bash
# Install in editable mode with test dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with debug logging
notebook-agent --task "..." --verbose
```

## Notes

NotEvolve is a research prototype. It executes model-generated code inside a
Jupyter kernel, so use an isolated environment and avoid exposing sensitive
files or credentials in the working directory.
