# notebook-agent

Training-free scaffolding that lets an agentic LLM work iteratively in a notebook-style environment with persistent execution state.

The LLM gets a live Jupyter kernel, a compact notebook view, and coding-agent-style file tools. It writes code cells, runs them, inspects outputs, and refines its approach over many rounds — without rerunning the entire notebook each time.

LLM backend is provider-flexible: Anthropic and OpenAI-compatible APIs are supported (set `--provider openai_compatible` and `--base-url` for custom endpoints).

## Quick start

```bash
# Install (into your conda/venv environment)
pip install -e ".[dev]"

# Run on a task
notebook-agent \
  --task "Find a discrete probability distribution maximizing P[X1+X2+X3 < 2X4]" \
  --notebook work.ipynb \
  --max-rounds 20
```

See [examples/beat_the_average_game/](examples/beat_the_average_game/) for a complete worked example.

## Architecture

The system is split into two layers:

- **Session layer** (`session.py`): A general-purpose framework for running one LLM session on a notebook. It handles the round loop, notebook rendering, tool dispatch, and LLM interaction. It knows nothing about campaigns, cross-session steering, or any specific problem domain.
- **Orchestrators** (`orchestrators/`): High-level consumers of the session layer. Each orchestrator is a different strategy for using notebook sessions to accomplish a goal. The built-in `CampaignRunner` implements an AlphaEvolve-style multi-session loop.

This separation means different orchestration strategies (single-shot, multi-session campaigns, population-based search, etc.) can all be built on the same session framework.

## Detailed workflow

This section traces how a single run flows through the codebase, module by module.

### 1. CLI bootstrap (`cli.py`)

`cli.py` is the entry point. It:

1. Parses CLI args and merges with `--config-file` JSON (CLI args take precedence via `ParameterSource` checks)
2. Resolves the API key (`resolve_api_key` tries provider-specific env vars: `ANTHROPIC_API_KEY` for anthropic, `OPENAI_API_KEY` for openai/openai_compatible, `LLM_API_KEY` as fallback)
3. Normalizes all filesystem paths (notebook, checkpoint dir, evaluator module) relative to `working_dir`
4. Builds an `AgentConfig` (pydantic model) and calls `run_agent(config)`

### 2. Component initialization (`cli.run_agent`)

`run_agent` wires everything together:

```
AgentConfig
  ├─► SecurityPolicy          (security/policy.py)
  ├─► KernelManager           (kernel/manager.py)     ← starts a real ipykernel subprocess
  │     └─► CellExecutor      (kernel/executor.py)
  ├─► NotebookStore           (persistence/store.py)   ← loads/creates the .ipynb file
  │     └─► Notebook           (models/notebook.py)    ← in-memory cell list
  ├─► NotebookManager         (notebook_ops/manager.py) ← all mutations go through here
  ├─► ContextRenderer         (context/renderer.py)
  ├─► CheckpointManager       (persistence/checkpoint.py)
  ├─► ToolRegistry            (tools/registry.py)      ← registers all 20+ tools
  │     ├─ notebook tools      (tools/notebook_tools.py)
  │     ├─ file tools          (tools/file_tools.py)
  │     ├─ context tools       (tools/context_tools.py)
  │     └─ evaluator tools     (tools/evaluator_tools.py) ← auto-discovered from module
  ├─► LLMInterface            (engine/llm_interface.py)
  └─► CampaignRunner          (orchestrators/campaign.py) ← orchestrates sessions
        └─► SessionRunner      (session.py)               ← drives one session's round loop
```

### 3. Campaign and session execution

The `CampaignRunner` (`orchestrators/campaign.py`) manages the high-level loop:

```
for session_num in 1..max_sessions:
    ┌──────────────────────────────────────────────────────────────┐
    │  1. Build task prompt                                        │
    │     CampaignRunner._build_task_prompt(task, session, history)│
    │     → wraps task in "## Your Task", appends session history  │
    │       for sessions 2+ (kernel restarted, notebook reloaded)  │
    │                                                              │
    │  2. Run session                                              │
    │     SessionRunner.run(task_prompt, rounds_per_session)        │
    │     → returns SessionResult (rounds_used, token counts, done)│
    │                                                              │
    │  3. Record summary for next session's context                │
    └──────────────────────────────────────────────────────────────┘
```

When `max_sessions=1` (the default), this reduces to a single session with `max_rounds` rounds — identical to the original single-loop behavior.

The `SessionRunner` (`session.py`) handles the round loop within each session:

```
for round_num in 1..max_rounds:
    ┌─────────────────────────────────────────────────────────────┐
    │  Step A: Build system prompt                                │
    │    SessionRunner._build_system_prompt(task_prompt)           │
    │    → task_prompt (from orchestrator) + SCAFFOLD_PROMPT       │
    │      (notebook conventions, tool list, strategy hints)       │
    │                                                             │
    │  Step B: Render notebook context                            │
    │    ContextRenderer.render(notebook, kernel_alive)           │
    │    → produces compact plain-text notebook view              │
    │                                                             │
    │  Step C: Send to LLM                                       │
    │    LLMInterface.chat(system=..., messages=[                 │
    │      { role: "user", content: notebook_context + round N }  │
    │    ], tools=tool_schemas)                                   │
    │                                                             │
    │  Step D: Tool call loop (within one round)                  │
    │    while LLM returns tool_calls:                            │
    │      for each tool_call:                                    │
    │        ToolRegistry.dispatch(name, args)                    │
    │          → SecurityPolicy.check_tool_call() first           │
    │          → then tool.execute(**args)                        │
    │      append results to messages                             │
    │      call LLM again with updated messages                   │
    │    (exits when LLM returns text-only or hits tool limit)    │
    │                                                             │
    │  Step E: Checkpoint                                         │
    │    CheckpointManager.save(round_number, notebook, messages) │
    │    → writes .checkpoints/round_0001_*.json + notebook copy  │
    └─────────────────────────────────────────────────────────────┘
```

Within each round, the LLM can make multiple tool calls in sequence. The tool call loop continues until the LLM responds with text only (no tool calls), or hits `max_tool_calls_per_round` (default 30), at which point it's forced to summarize.

### 4. How the notebook is presented to the LLM

The notebook is rendered as **structured plain text** (not JSON), inspired by the "brief format" in [notebookllm_mcp](https://github.com/nicholasgasior/notebookllm-mcp). This is produced by `ContextRenderer` (`context/renderer.py`).

#### System prompt (sent once per round)

The system prompt is composed of two parts:

1. **Task prompt** (from the orchestrator): includes the task description, `system_prompt_extra`, and any cross-session context
2. **Scaffold prompt** (`SCAFFOLD_PROMPT` in `session.py`): task-agnostic instructions about the notebook environment, available tools, cell state conventions, and strategy

```
## Your Task
{task description from --task}

{system_prompt_extra from config, if any}

{session history, for sessions 2+ in multi-session mode}

You are an expert research agent working in a notebook environment.

## Environment
You have a persistent Jupyter notebook with a live Python kernel.
Objects and variables persist across cell executions within a round
and across rounds. You do NOT need to re-import or re-define things
that are already in the kernel's memory.

## Available Tools
You have tools for:
- Notebook reading: read_notebook, read_cell, expand_output
- Notebook writing: add_cell (code/markdown/title), edit_cell, delete_cell
- Execution: run_cell, run_from_cell, run_stale_cells
- Context management: fold_cell, unfold_cell
- File operations: glob_files, grep_files, read_file, list_tree
- Kernel management: restart_kernel, create_savepoint, list_savepoints, restore_savepoint
- Evaluator: {auto-discovered evaluator tool names + descriptions}

## Notebook Conventions
- Cell states: CLEAN (executed, up-to-date), DIRTY (edited, needs re-run),
  STALE (upstream changed), UNEXECUTED (never run)
- When you edit a code cell, it becomes DIRTY and downstream code cells become STALE
- Always run cells after editing to update the kernel state
- Long outputs are cropped by default; use expand_output() to see full output
- Earlier cells are folded (compact view); use read_cell() or unfold_cell()

## Strategy
- Work iteratively: write code, run it, inspect results, refine
- Use markdown cells for notes and observations
- Create savepoints before major changes
```

#### User message (the notebook context, sent each round)

```
## Current Notebook State

=== NOTEBOOK (6 cells, kernel: alive) ===

[0] MARKDOWN | # Data Pipeline
[1] CODE [CLEAN] exec:1 | import pandas as pd  (3 lines)  [output: 5 lines]
[2] CODE [STALE] exec:2 | def process(df):  (12 lines)  [has error]

--- [3] CODE [DIRTY] ---
def process(df):
    result = df.groupby("category").sum()
    return result
[Source changed, not yet re-executed]
--- stale output ---
   category  value
0  A         42

--- [4] CODE [CLEAN] exec:4 ---
process(df)
--- output ---
   category  value
0  A         100
1  B         200
... [47 more lines. Use expand_output(4) to see full]
0  Z         999

--- [5] CODE [UNEXECUTED] ---
# TODO: add filtering
[Not yet executed]

## Round 3
Continue working on the task. Use the available tools to make progress.
```

Key rendering rules (`ContextRenderer`):

| Rule | Detail |
|------|--------|
| **Folding** | Cells before index `total - unfold_last_n` are folded to one line: `[i] TYPE [STATE] exec:N \| first_line (N lines)` |
| **Unfolding** | Last N cells (default 3) show full source + outputs |
| **Override** | LLM can call `fold_cell(i)` / `unfold_cell(i)` to override defaults; explicit fold/unfold sticks across rounds |
| **Output cropping** | Outputs longer than `max_output_lines` (default 30) are split: first half + `... [N more lines. Use expand_output(i) to see full]` + last half |
| **State tags** | Code cells show `[CLEAN]`, `[DIRTY]`, `[STALE]`, or `[UNEXECUTED]` |
| **Stale outputs** | If cell is DIRTY: `--- stale output ---`. If STALE: `--- output [STALE] ---` |
| **Output hints** | Folded cells with output show `[output: N lines]` or `[has error]` |

### 5. Tool dispatch and security (`tools/registry.py`, `security/policy.py`)

When the LLM calls a tool:

1. `ToolRegistry.dispatch(name, args)` looks up the tool
2. `SecurityPolicy.check_tool_call(name, args)` validates:
   - File path arguments stay inside `working_dir` (workspace sandbox)
   - File path arguments don't point to protected evaluator paths
3. The tool's `.execute(**args)` runs. For execution tools (`run_cell` etc.), `SecurityPolicy.check_code()` additionally scans the cell source for forbidden patterns (`pip install`, `subprocess.`, `os.system(`)
4. Result returned as `ToolResult(content, is_error)` to the LLM

### 6. Cell execution (`kernel/executor.py`)

When `run_cell` is called:

1. `NotebookManager.run_cell(index)` delegates to `CellExecutor.execute(source)`
2. `CellExecutor` calls `jupyter_client`'s `execute_interactive()` with a custom `_output_hook` callback (pattern from jupyter-server-nbmodel)
3. The hook collects streaming outputs: `display_data`, `stream` (merged consecutively), `execute_result`, `error`
4. Raw outputs are parsed into `CellOutput` objects (ANSI escape codes stripped from errors)
5. Cell state updated: success → `CLEAN`, error → keeps DIRTY/STALE but stores error output
6. `StalenessTracker.on_execute_cell()` marks the cell CLEAN

### 7. Staleness propagation (`notebook_ops/staleness.py`)

Linear, position-based (no dependency graph):

| Event | Effect |
|-------|--------|
| **Edit code cell at index i** | Cell i → DIRTY; all code cells at j > i → STALE |
| **Delete code cell at index i** | All code cells at j > i → STALE |
| **Execute cell at index i** | Cell i → CLEAN (stores `last_executed_source`) |
| **Edit markdown cell** | No staleness changes |

### 8. Evaluator auto-discovery (`tools/evaluator_tools.py`)

When `--evaluator-module` is set:

1. `_load_module()` loads the module — either by file path (`.py`) or import path (`pkg.mod`)
2. `_autodiscovered_specs()` scans for public functions starting with `evaluate` or `check`
3. For each function, `_schema_from_annotation()` derives a JSON Schema from Python type hints (uses `typing.get_type_hints()` to resolve string annotations from `from __future__ import annotations`)
4. Each function is wrapped in an `EvaluatorTool` and registered in the `ToolRegistry`

The LLM sees the evaluator as callable tools but **cannot read the evaluator source** — `SecurityPolicy` blocks file access to evaluator paths.

### 9. Checkpointing (`persistence/checkpoint.py`, `persistence/store.py`)

After each round:

- `CheckpointManager.save()` writes `{checkpoint_dir}/round_NNNN_{timestamp}.json` containing the round number, messages, and tool call log
- The notebook `.ipynb` is also saved via `NotebookStore.save()` (standard nbformat v4)
- Savepoints (user-triggered via `create_savepoint` tool) are separate: full notebook snapshots stored under `{checkpoint_dir}/savepoints/` with an `index.json` manifest

## Cell states and staleness

```
UNEXECUTED ─── (run) ──► CLEAN
CLEAN ──── (edit source) ──► DIRTY ── (run) ──► CLEAN
CLEAN ──── (upstream edit/delete) ──► STALE ── (run) ──► CLEAN
```

When you edit a code cell, it becomes **DIRTY** and all downstream code cells become **STALE**. Markdown edits never affect execution state.

## Tools available to the LLM

| Category | Tools |
|----------|-------|
| **Notebook reading** | `read_notebook`, `read_cell`, `expand_output` |
| **Notebook writing** | `add_cell` (supports `code`/`markdown`/`title`), `edit_cell`, `delete_cell` |
| **Execution** | `run_cell`, `run_from_cell`, `run_stale_cells` |
| **Context** | `fold_cell`, `unfold_cell` |
| **File ops** | `glob_files`, `grep_files`, `read_file`, `list_tree` |
| **Kernel** | `restart_kernel`, `create_savepoint`, `list_savepoints`, `restore_savepoint` |
| **Evaluator** | Auto-discovered from `--evaluator-module` (source hidden from LLM) |

## Security

- **pip install** blocked in all forms (`pip install`, `!pip`, `%pip`, `subprocess.`, `os.system(`)
- **Evaluator isolation** — evaluator source is never readable by the LLM; it can only call evaluator tools and see scores
- **Workspace sandbox** — file read/search tools are restricted to `working_dir`; paths outside are rejected
- **Path protection** — evaluator file paths and directories are blocked from all file tools

## CLI reference

```
notebook-agent [OPTIONS]

Options:
  -t, --task TEXT              Task description for the agent (required)
  -n, --notebook PATH          Notebook file path [default: notebook.ipynb]
  -m, --model TEXT             LLM model name [default: claude-sonnet-4-20250514]
  -p, --provider TEXT          LLM provider: anthropic, openai, openai_compatible [default: anthropic]
  --api-key TEXT               API key (or use OPENAI_API_KEY/ANTHROPIC_API_KEY/LLM_API_KEY)
  --base-url TEXT              Optional base URL for OpenAI-compatible APIs
  --max-rounds INT             Maximum rounds [default: 50]
  --max-sessions INT           Number of sessions (1 = single session) [default: 1]
  --rounds-per-session INT     Rounds per session (0 = use max-rounds) [default: 0]
  --max-tokens INT             Max tokens per LLM response [default: 16384]
  --thinking-budget INT        Extended thinking budget [default: 10000]
  -w, --working-dir PATH       Working directory [default: .]
  --checkpoint-dir PATH        Checkpoint directory [default: .checkpoints]
  --evaluator-path PATH        Protected evaluator paths (repeatable)
  --evaluator-module TEXT      Python module/file with evaluator functions
  --kernel TEXT                Jupyter kernel name [default: python3]
  -v, --verbose                Debug logging
  -c, --config-file PATH       JSON config file
```

### Config file

All CLI options can also be set in a JSON config file (`--config-file`). CLI args override config file values when explicitly provided. Config file keys use the `AgentConfig` field names:

```json
{
  "task": "Find optimal distribution...",
  "llm_model": "claude-sonnet-4-20250514",
  "llm_provider": "anthropic",
  "llm_base_url": null,
  "llm_extra_headers": {},
  "llm_temperature": 0.0,
  "llm_thinking_budget": 10000,
  "notebook_path": "notebook.ipynb",
  "working_dir": ".",
  "evaluator_module": "evaluator.py",
  "evaluator_paths": ["evaluator.py"],
  "unfold_last_n": 3,
  "max_output_lines": 30,
  "max_rounds": 50,
  "max_tool_calls_per_round": 30,
  "max_sessions": 1,
  "rounds_per_session": 0,
  "system_prompt_extra": ""
}
```

## Project structure

```
notebook_agent/
├── session.py       SessionRunner + SCAFFOLD_PROMPT (the core session framework)
├── orchestrators/   High-level consumers of the session layer
│   └── campaign.py    CampaignRunner (multi-session loop)
├── models/          Cell, CellOutput, Notebook data structures
│   ├── cell.py        Cell, CellOutput, CellType, CellState
│   └── notebook.py    Notebook (cell list + metadata + helpers)
├── kernel/          Jupyter kernel lifecycle + cell execution
│   ├── manager.py     KernelManager (start/stop/restart ipykernel subprocess)
│   └── executor.py    CellExecutor (execute_interactive + output hook)
├── notebook_ops/    Notebook CRUD, staleness propagation, savepoints
│   ├── manager.py     NotebookManager (insert/edit/delete/run + savepoints)
│   └── staleness.py   StalenessTracker (linear downstream propagation)
├── context/         Compact notebook rendering for LLM context
│   └── renderer.py    ContextRenderer (fold/unfold/crop → plain text)
├── tools/           All LLM-facing tools (notebook, file, evaluator)
│   ├── base.py        BaseTool ABC + ToolResult
│   ├── registry.py    ToolRegistry (dispatch + security gate)
│   ├── notebook_tools.py  13 notebook/kernel/savepoint tools
│   ├── file_tools.py      4 file operation tools
│   ├── context_tools.py   2 fold/unfold tools
│   └── evaluator_tools.py Auto-discovery + EvaluatorTool wrapper
├── engine/          LLM API adapter
│   └── llm_interface.py   LLMInterface (Anthropic + OpenAI adapters)
├── security/        Code scanning, path protection, evaluator isolation
│   └── policy.py      SecurityPolicy (check_code, check_path, check_tool_call)
├── persistence/     .ipynb serialization + per-round checkpoints
│   ├── store.py        NotebookStore (nbformat v4 read/write)
│   └── checkpoint.py   CheckpointManager (round snapshots)
├── config.py        AgentConfig (pydantic model, all settings)
└── cli.py           CLI entry point (click + config merging)
```

## Development

```bash
# Run tests (54 tests across 7 files)
pytest tests/ -v

# Run with debug logging
notebook-agent --task "..." --verbose
```
