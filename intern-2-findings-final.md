# Consolidated Findings (intern-1 + intern-2 agreed)

Issues where both reviewers independently flagged the same problem, or where one flagged it and the other confirmed it on cross-review. Ordered by severity.

---

## High

### 1. Text-only LLM response prematurely terminates the session

*Both independently found.*

`SessionRunner._run_round` returns `response.stop_reason == "end_turn" and not response.tool_calls` to signal session completion. For Anthropic, any text-only response naturally has `stop_reason == "end_turn"`. For OpenAI, the `stop_reason_map` at `llm_interface.py:266` explicitly maps `finish_reason="stop"` to `"end_turn"`. So on both providers, the first round where the model responds with text instead of tool calls ends the entire session — defeating the multi-round loop.

The model has no way to share an observation or ask a question without triggering session termination. `HumanLLM` avoids this by using `"stop"` for text-only and `"end_turn"` only for explicit `/done`, but the real providers don't have that distinction.

**Files:** `session.py:244`, `llm_interface.py:266-270`

### 2. Relative file paths resolved against process CWD, not `working_dir`

*Both independently found. Intern-2 adds the security-check angle.*

`GlobFilesTool`, `GrepFilesTool`, and `ListTreeTool` resolve relative paths with `Path(path).resolve()`, which uses the process CWD. `ReadFileTool` correctly uses `self._working_dir / fpath`. When CWD differs from `--working-dir`, the first three tools search the wrong directory.

Additionally, `SecurityPolicy.check_path()` also resolves against CWD before checking containment in `working_dir`. This means the security check and the tool logic are both wrong in the same way — valid workspace-relative paths may be rejected, and CWD-relative paths outside the workspace may pass the check depending on CWD.

**Files:** `file_tools.py:31,75,218` (tools), `security/policy.py:73` (check_path), vs. `file_tools.py:160-162` (ReadFileTool — correct)

### 3. Security boundary is pattern-based only; kernel code runs unrestricted

*Intern-2 found. Intern-1 confirms, and independently flagged a related sub-issue (evaluator name substring matching too broad).*

`SecurityPolicy.check_code()` scans source text for a short list of regexes and evaluator path/name substrings. After that, the kernel executes unrestricted Python. Notebook code can trivially bypass the sandbox with normal filesystem APIs (e.g., `pathlib.Path("/etc/passwd").read_text()`), and evaluator isolation can be bypassed with computed paths or indirect file discovery.

Separately, the evaluator name check uses plain substring matching (`name in lowered`), which causes false positives with common filenames — e.g., an evaluator named `data.py` would block any code containing that substring.

**Files:** `security/policy.py:52-69`, `notebook_tools.py:202-208`

---

## Medium

### 4. Inserting a code cell upstream does not stale downstream cells

*Intern-2 found. Intern-1 confirms.*

`insert_cell()` mutates the cell list and bumps the version, but never calls any `StalenessTracker` method. Since staleness is position-based, inserting a code cell before already-executed cells should invalidate downstream execution state. Currently those cells remain `CLEAN`, even though the kernel has never seen the new upstream cell's effects.

Compare with `edit_cell()` and `delete_cell()`, which both propagate staleness correctly.

**File:** `notebook_ops/manager.py:110-131`

### 5. Restoring a savepoint does not restore or reset kernel state

*Intern-2 found. Intern-1 confirms.*

Savepoints snapshot notebook JSON only. `restore_savepoint()` replaces notebook cells from the snapshot, but the live kernel namespace is untouched. After a restore, the notebook shows an older cell state while the kernel still holds newer variables. Subsequent execution uses the stale kernel state, leading to silent semantic mismatches.

**File:** `notebook_ops/manager.py:240-249`

### 6. Tool-call-limit path drops the wrap-up response from checkpoints and usage accounting

*Intern-2 found. Intern-1 confirms.*

When `max_tool_calls_per_round` is hit, the code makes a second LLM call (`final_response`) asking for a summary. But:
- `final_response` is not appended to `messages`, so the checkpoint is incomplete.
- Token usage from `final_response` is not added to `_total_usage`.
- The done/not-done check on line 244 still uses the earlier `response` variable (which had tool calls), not `final_response`. This happens to return `False` by accident (since `response.tool_calls` is non-empty), but the logic is using stale data.

**File:** `session.py:214-244`

---

## Low

### 7. Notebook reload produces contradictory state markers

*Intern-2 found. Intern-1 confirms.*

On load, a code cell without outputs is reset to `UNEXECUTED` (line 98), but its `execution_count` is preserved from the saved file (line 110). The renderer then shows `[i] CODE [UNEXECUTED] exec:5`, which is contradictory and misleading for the model — a cell cannot be both unexecuted and have an execution count.

**Files:** `persistence/store.py:93-110`
