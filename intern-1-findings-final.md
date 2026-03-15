# intern-1 findings final

This file captures the issues that both `intern-1-findings.md` and `intern-2-findings.md` independently support after reconciling scope and wording.

## Agreed issues

### 1. Relative path handling is inconsistent in `check_path`, `glob_files`, and `grep_files`

Both reviews identify a real path-resolution bug in the file access flow.

- `SecurityPolicy.check_path()` resolves relative paths directly with `Path(path).resolve()`, which uses the process CWD.
- `GlobFilesTool.execute()` and `GrepFilesTool.execute()` do the same before normalizing paths against `working_dir`.
- `ToolRegistry` runs `check_tool_call()` before per-tool path normalization, so valid workspace-relative paths can be rejected or searched in the wrong place when `--working-dir` differs from the launcher CWD.

Clarification:
- This issue does not include `ListTreeTool`. That tool already rebases relative paths onto `self._working_dir` before resolving them.

References:
- `notebook_agent/security/policy.py`
- `notebook_agent/tools/registry.py`
- `notebook_agent/tools/file_tools.py`

### 2. Stop-reason handling can terminate the session too early

Both reviews identify a correctness problem in the session lifecycle.

- `SessionRunner` treats a text-only response with `stop_reason == "end_turn"` as meaning the whole session is complete.
- The OpenAI adapter maps normal `finish_reason="stop"` responses to `stop_reason="end_turn"`.
- That means an ordinary text-only model turn can end the entire run instead of just ending the current round.

Impact:
- Multi-round runs can stop after the first successful round even when the model has not actually completed the task.

References:
- `notebook_agent/session.py`
- `notebook_agent/engine/llm_interface.py`

## Note

These are the only issues that both reviews clearly converged on. Intern-1 surfaced several additional plausible issues, but they were not independently identified in both reviews, and one part of the path-resolution item needed narrowing as noted above.
