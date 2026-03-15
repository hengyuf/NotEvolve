# intern-2 findings

Read-only review of the `notebook_agent` project. I did not run tests or make any code changes while producing these notes. I treated `references/` as out of scope and reviewed the project code under `notebook_agent/` plus the current tests.

## Findings

### 1. High: normal provider runs appear to stop after the first completed round

`SessionRunner` ends a round as soon as the model returns a response with no tool calls, then treats `stop_reason == "end_turn"` as "the agent is done". For OpenAI responses, ordinary text completions map `finish_reason="stop"` to `end_turn`, so the common case becomes: finish round -> mark session done -> exit early. That defeats the intended multi-round loop.

References:
- `/home/baihe/notebook-agent/notebook_agent/session.py#L167`
- `/home/baihe/notebook-agent/notebook_agent/session.py#L244`
- `/home/baihe/notebook-agent/notebook_agent/engine/llm_interface.py#L266`

### 2. High: the security boundary is not real for executed notebook code

`SecurityPolicy.check_code()` only scans the source text for a few regexes and direct evaluator-path/name matches. After that, the execution tools run unrestricted Python in a real kernel. That means notebook code can still use normal Python filesystem APIs to read outside `working_dir`, and evaluator isolation can likely be bypassed with computed paths or indirect discovery.

References:
- `/home/baihe/notebook-agent/notebook_agent/security/policy.py#L52`
- `/home/baihe/notebook-agent/notebook_agent/security/policy.py#L71`
- `/home/baihe/notebook-agent/notebook_agent/tools/notebook_tools.py#L202`

### 3. High: relative file paths are resolved against the process CWD, not `working_dir`

The registry runs `check_tool_call()` before tools normalize paths. `check_path()` uses `Path(path).resolve()` directly, and `glob_files` / `grep_files` do the same. If the CLI is launched from one directory but `--working-dir` points somewhere else, valid relative paths will be rejected or searched in the wrong place.

References:
- `/home/baihe/notebook-agent/notebook_agent/tools/registry.py#L43`
- `/home/baihe/notebook-agent/notebook_agent/security/policy.py#L71`
- `/home/baihe/notebook-agent/notebook_agent/tools/file_tools.py#L30`
- `/home/baihe/notebook-agent/notebook_agent/tools/file_tools.py#L74`

### 4. Medium: inserting a new code cell upstream does not stale downstream executed cells

`insert_cell()` only mutates the cell list and bumps the notebook version. Since the rest of the system treats staleness as position-based, adding a code cell before already-executed cells should invalidate downstream execution state, but currently those cells remain `CLEAN`.

References:
- `/home/baihe/notebook-agent/notebook_agent/notebook_ops/manager.py#L110`

### 5. Medium: restoring a savepoint does not restore or reset kernel state

Savepoints snapshot notebook JSON only. `restore_savepoint()` reloads notebook cells from disk, but it leaves the live kernel namespace untouched. After a restore, the notebook view can represent an older state while the kernel still contains newer variables/imports, which makes subsequent execution semantics unreliable.

References:
- `/home/baihe/notebook-agent/notebook_agent/notebook_ops/manager.py#L223`
- `/home/baihe/notebook-agent/notebook_agent/notebook_ops/manager.py#L240`

### 6. Medium: the tool-call-limit path drops the forced wrap-up response from checkpoints and usage accounting

When `max_tool_calls_per_round` is hit, the code asks the model for a final summary, but that `final_response` is only logged. It is not appended to `messages`, not added to token usage, and the final done/not-done decision still uses the earlier tool-calling response.

References:
- `/home/baihe/notebook-agent/notebook_agent/session.py#L214`
- `/home/baihe/notebook-agent/notebook_agent/session.py#L226`
- `/home/baihe/notebook-agent/notebook_agent/session.py#L244`

### 7. Low: notebook reload can produce contradictory state markers

On load, no-output code cells are reset to `UNEXECUTED`, but their old `execution_count` is preserved. That can produce notebook context that says a cell is both unexecuted and `exec:N`, which is misleading for the model.

References:
- `/home/baihe/notebook-agent/notebook_agent/persistence/store.py#L93`
- `/home/baihe/notebook-agent/notebook_agent/persistence/store.py#L110`

## Coverage gaps

I did not find tests covering `SessionRunner`, `CampaignRunner`, CLI `working_dir` behavior, or savepoint/kernel consistency. Those gaps line up with several of the issues above.
