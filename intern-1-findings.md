# Code Review Findings

Read-only audit of the `notebook_agent` codebase. Organized by severity.

---

## Bugs / Correctness Issues

### 1. Relative path resolution inconsistency in file tools

**Files:** `file_tools.py:31,75,218`

`GlobFilesTool`, `GrepFilesTool`, and `ListTreeTool` resolve relative paths against **process CWD** instead of `working_dir`:

```python
# GlobFilesTool:31 — resolves against CWD
base = Path(path).resolve() if path else self._working_dir

# ReadFileTool:160-162 — correctly resolves against working_dir
fpath = Path(path)
if not fpath.is_absolute():
    fpath = self._working_dir / fpath
```

If the process CWD differs from `working_dir` (common when using `--working-dir`), relative paths in glob/grep/tree will search the wrong directory. The security check catches out-of-sandbox access but the behavior is still wrong.

### 2. `ListTreeTool.EXCLUDE_DIRS` glob pattern never matches

**File:** `file_tools.py:208,248`

```python
EXCLUDE_DIRS = {
    "__pycache__", ".git", "node_modules", ..., "*.egg-info", ...
}
# Used as exact string match:
entries = [e for e in entries if e.name not in self.EXCLUDE_DIRS ...]
```

`"*.egg-info"` is a glob pattern but the check does exact `name not in set`. A directory named `mypackage.egg-info` will never match `"*.egg-info"`. This entry is dead.

Additionally: `.git`, `.tox`, `.mypy_cache`, `.pytest_cache`, `.ipynb_checkpoints` are redundant in the set because the same line also filters `e.name.startswith(".")` — all dot-prefixed names are already excluded.

### 3. Cell ordering broken when using explicit unfold

**File:** `renderer.py:45-69`

The renderer accumulates folded and unfolded cells into separate lists, then emits all folded cells first, then all unfolded cells:

```python
folded_section: list[str] = []
unfolded_section: list[str] = []
# ...
parts.extend(folded_section)
parts.extend(unfolded_section)
```

With default behavior (last N unfolded), this preserves order because all unfolded cells are at the tail. But if a user explicitly unfolds cell 3 in a 10-cell notebook with `unfold_last_n=2`, the output order is: `[0,1,2,4,5,6,7,  3,8,9]` — cell 3 appears after cell 7. The LLM sees cells out of order.

### 4. `RunCellTool` duplicates error output

**File:** `notebook_tools.py:216-222`

```python
for out in result.outputs:
    if out.content:
        parts.append(out.content)       # error traceback appears here

if result.error:
    parts.append(f"\nError:\n{result.error}")  # ...and again here
```

`result.error` is built from error-type outputs in `executor.py:101-105` — the same content already iterated above. The LLM sees the full traceback twice.

### 5. `_rg_search` conflates all non-0/1 return codes with "rg not found"

**File:** `file_tools.py:99-100`

```python
if result.returncode != 0:
    raise FileNotFoundError("rg not found")
```

Ripgrep returns 2 for regex syntax errors, other codes for other failures. These are silently swallowed and fall through to the Python regex search, which may also fail differently. Real rg errors are hidden.

---

## Design Issues

### 6. Synchronous API calls block the event loop

**File:** `llm_interface.py:126,244`

`_ensure_client` creates `anthropic.Anthropic` (sync) and `openai.OpenAI` (sync), then calls them inside `async` methods:

```python
response = self._client.messages.create(**kwargs)   # blocking
response = self._client.chat.completions.create(**kwargs)  # blocking
```

This blocks the event loop. It works today because nothing else is concurrent, but it would silently serialize any future concurrent work. Should use `AsyncAnthropic`/`AsyncOpenAI`, or wrap in `asyncio.to_thread()` (as `HumanLLM` does for `input()`).

### 7. Text-only response always signals session completion

**File:** `session.py:244`

```python
return response.stop_reason == "end_turn" and not response.tool_calls
```

For Anthropic, any text-only response has `stop_reason == "end_turn"`. If the model responds with an observation or question without tool calls, the session terminates. The model has no way to say "I want to share a thought, then continue working." The `HumanLLM` avoids this by using `stop_reason == "stop"` for text-only and `"end_turn"` only for explicit `/done`.

### 8. Evaluator file name substring matching is overly broad

**File:** `security/policy.py:54-61`

```python
if name and name in lowered:
    raise SecurityError(...)
```

If the evaluator file is `score.py`, the filename `score.py` is matched as a substring of the lowered code. Something like `"high_score.pyformat(...)"` would be blocked. More importantly, this is a plain string match — not a word-boundary or path-boundary match. With common filenames, false positives are likely.

### 9. Multiple text content blocks: last one wins

**File:** `llm_interface.py:134-135`

```python
if block.type == "text":
    content_text = block.text  # overwrites previous
```

If the Anthropic response contains multiple text blocks (possible with some model configurations), only the last one is kept. Earlier text content is silently lost.

---

## Code Quality Issues

### 10. `ANSI_ESCAPE` defined but never used

**File:** `executor.py:19`

```python
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")  # never referenced
```

Dead code. ANSI stripping actually happens in `cell.py:56-58`.

### 11. ANSI regex compiled inside a method on every call

**File:** `cell.py:56-58`

```python
@staticmethod
def from_nbformat(output: dict) -> CellOutput:
    # ... inside the error branch:
    import re
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
```

The `import re` is fine (cached by Python), but `re.compile()` is called every time an error output is parsed. Should be a module-level constant. Also, this regex is narrower than the one in `executor.py` (only matches `m` terminator vs any letter) — ANSI sequences not ending in `m` (e.g. cursor movement, screen clear) won't be stripped from error tracebacks.

### 12. Three tool classes access private `NotebookManager._bump()`

**Files:** `notebook_tools.py:323`, `context_tools.py:30,53`

`RestartKernelTool`, `FoldCellTool`, and `UnfoldCellTool` all call `self._nb._bump()` — a private method. This violates encapsulation. There should be a public method for "mark notebook as modified."

### 13. Unused `stale` parameter in `_render_output`

**File:** `renderer.py:125`

```python
def _render_output(self, output: CellOutput, cell_index: int, stale: bool = False) -> str:
    # `stale` is never used in the method body
```

### 14. Unused variable `old_source` in `edit_cell`

**File:** `manager.py:139`

```python
old_source = cell.source  # assigned but never read
```

### 15. Wasteful Cell creation for ID generation

**File:** `store.py:105`

```python
id=nb_cell.get("id", None) or Cell().id
```

Creates an entire `Cell` dataclass instance just to get a UUID. `str(uuid.uuid4())` would suffice.

### 16. `_sanitize_messages` only truncates string content

**File:** `checkpoint.py:106-108`

List-type content (tool_use blocks, tool_result blocks with large outputs) is never truncated. Large tool results can produce unbounded checkpoint files.

### 17. `import json` at end of test file

**File:** `test_human_llm.py:191`

```python
import json  # after all test classes, at bottom of file
```

This import is used in `TestMergeMultilineTools.test_multiline_json` (line 142). It works because Python resolves names at call time, not definition time, but it's unusual and looks like an afterthought.

### 18. Inconsistent output truncation policies

**File:** `notebook_tools.py:261-264`

`RunFromCellTool` uses a hardcoded threshold of 20 lines for truncation, while the renderer's `max_output_lines` is configurable (default 30). Two different truncation policies for the same kind of content.
