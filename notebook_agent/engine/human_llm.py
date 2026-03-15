"""Human-in-the-loop LLM replacement for interactive testing.

Instead of calling an LLM API, this module displays prompts on the terminal
and reads operator input, parsing it into LLMResponse objects. This allows
testing the full notebook-agent scaffolding without API calls.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from notebook_agent.engine.llm_interface import LLMInterface, LLMResponse, ToolCall


class _Colors:
    """ANSI color codes, disabled when stdout is not a TTY."""

    def __init__(self) -> None:
        enabled = sys.stdout.isatty()
        self.BOLD = "\033[1m" if enabled else ""
        self.DIM = "\033[2m" if enabled else ""
        self.CYAN = "\033[36m" if enabled else ""
        self.GREEN = "\033[32m" if enabled else ""
        self.YELLOW = "\033[33m" if enabled else ""
        self.RED = "\033[31m" if enabled else ""
        self.MAGENTA = "\033[35m" if enabled else ""
        self.RESET = "\033[0m" if enabled else ""


class HumanLLM(LLMInterface):
    """LLM replacement that uses the human operator as the language model.

    Displays the full prompt (system, messages, tools) to the terminal and
    collects the operator's response, parsing it into an LLMResponse.

    Usage::

        notebook-agent --provider human --task "your task" --notebook nb.ipynb
    """

    def __init__(self) -> None:
        super().__init__(provider="human", model="human", api_key="")
        self._call_count = 0
        self._tc_counter = 0
        self._last_system: str | None = None
        self._c = _Colors()
        # Set by chat() before _collect_input runs in a thread
        self._current_tools: list[dict] | None = None
        self._current_system: str | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Display prompt to the operator and collect a response."""
        self._call_count += 1
        self._current_tools = tools
        self._current_system = system

        # Separator
        c = self._c
        print(f"\n{c.DIM}{'=' * 64}{c.RESET}")
        print(f"{c.BOLD}  LLM Call #{self._call_count}{c.RESET}")
        print(f"{c.DIM}{'=' * 64}{c.RESET}")

        # System prompt (only when changed)
        if system and system != self._last_system:
            self._display_system(system)
            self._last_system = system

        # Messages
        self._display_messages(messages)

        # Tools (compact)
        if tools:
            self._display_tools_compact(tools)

        # Collect input (blocking, wrapped for asyncio)
        lines = await asyncio.to_thread(self._collect_input)

        # Parse into LLMResponse
        return self._parse_response(lines)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _display_system(self, system: str) -> None:
        c = self._c
        print(f"\n{c.CYAN}{c.BOLD}-- SYSTEM PROMPT {'—' * 46}{c.RESET}")
        print(f"{c.CYAN}{system}{c.RESET}")
        print(f"{c.CYAN}{'—' * 64}{c.RESET}")

    def _display_messages(self, messages: list[dict]) -> None:
        c = self._c
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content")

            if role == "user" and isinstance(content, str):
                print(f"\n{c.GREEN}{c.BOLD}-- USER {'—' * 56}{c.RESET}")
                print(f"{c.GREEN}{content}{c.RESET}")

            elif role == "assistant" and isinstance(content, list):
                print(f"\n{c.MAGENTA}{c.BOLD}-- ASSISTANT {'—' * 51}{c.RESET}")
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        print(f"{c.MAGENTA}{block.get('text', '')}{c.RESET}")
                    elif btype == "tool_use":
                        args_str = json.dumps(block.get("input", {}))
                        print(
                            f"{c.MAGENTA}  [tool_use] {block.get('name')}"
                            f"({args_str}){c.RESET}"
                        )

            elif role == "user" and isinstance(content, list):
                # Tool results
                print(f"\n{c.YELLOW}{c.BOLD}-- TOOL RESULTS {'—' * 48}{c.RESET}")
                for block in content:
                    if block.get("type") == "tool_result":
                        is_err = block.get("is_error", False)
                        color = c.RED if is_err else c.YELLOW
                        label = "ERROR" if is_err else "OK"
                        tid = block.get("tool_use_id", "?")
                        raw = block.get("content", "")
                        # Truncate long results
                        lines = raw.split("\n") if isinstance(raw, str) else [str(raw)]
                        max_lines = 50
                        if len(lines) > max_lines:
                            shown = "\n".join(lines[:max_lines])
                            print(
                                f"{color}  [{label}] {tid}:\n{shown}\n"
                                f"  [... {len(lines) - max_lines} more lines]{c.RESET}"
                            )
                        else:
                            print(f"{color}  [{label}] {tid}:\n{raw}{c.RESET}")

            elif role == "user":
                # Fallback for plain string or other content
                print(f"\n{c.GREEN}{c.BOLD}-- USER {'—' * 56}{c.RESET}")
                print(f"{c.GREEN}{content}{c.RESET}")

    def _display_tools_compact(self, tools: list[dict]) -> None:
        c = self._c
        print(f"\n{c.DIM}-- Available Tools {'—' * 45}{c.RESET}")
        for t in tools:
            name = t.get("name", "?")
            desc = t.get("description", "")
            # Truncate description
            if len(desc) > 60:
                desc = desc[:57] + "..."
            schema = t.get("input_schema", {})
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            params = []
            for pname in props:
                params.append(f"{pname}*" if pname in required else pname)
            param_str = ", ".join(params)
            print(f"{c.DIM}  {name}({param_str}) - {desc}{c.RESET}")
        print(f"{c.DIM}{'—' * 64}{c.RESET}")

    def _display_tools_full(self, tools: list[dict]) -> None:
        c = self._c
        print(f"\n{c.BOLD}-- Tool Details {'—' * 49}{c.RESET}")
        for t in tools:
            name = t.get("name", "?")
            desc = t.get("description", "")
            schema = t.get("input_schema", {})
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            print(f"\n  {c.BOLD}{name}{c.RESET}: {desc}")
            for pname, pschema in props.items():
                req_tag = " (required)" if pname in required else ""
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                print(f"    {pname}: {ptype}{req_tag} - {pdesc}")
        print()

    def _print_help(self) -> None:
        c = self._c
        print(f"""
{c.BOLD}Input Syntax:{c.RESET}
  Plain text            Text content of the LLM response
  /tool name {{json}}     Tool call (multiple allowed)
  /tool name            Tool call with empty args
  /done                 End session (stop_reason=end_turn)
  (blank line)          Submit response

{c.BOLD}Commands (not added to response):{c.RESET}
  /help                 Show this help
  /tools                Show full tool descriptions
  /system               Re-display system prompt

{c.BOLD}Multi-line JSON:{c.RESET}
  /tool add_cell {{
    "cell_type": "code",
    "source": "x = 1\\ny = 2"
  }}
  (blank line only submits when all braces are closed)

{c.BOLD}Round/Session control:{c.RESET}
  Text-only response    Ends the current round, continues session
  /done                 Ends the current round AND the session
  Tool calls            Continues the tool loop within the round
""")

    # ------------------------------------------------------------------
    # Input collection
    # ------------------------------------------------------------------

    def _collect_input(self) -> list[str]:
        """Collect multi-line input from the operator. Blocking."""
        c = self._c
        print(
            f"\n{c.BOLD}Your response "
            f"(blank line to submit, /help for commands):{c.RESET}"
        )

        lines: list[str] = []
        brace_depth = 0

        while True:
            try:
                line = input(f"{c.DIM}>{c.RESET} ")
            except EOFError:
                break

            stripped = line.strip()

            # Inline commands (don't add to response)
            if stripped == "/help" and brace_depth == 0:
                self._print_help()
                continue
            if stripped == "/tools" and brace_depth == 0:
                if self._current_tools:
                    self._display_tools_full(self._current_tools)
                else:
                    print("  (no tools available)")
                continue
            if stripped == "/system" and brace_depth == 0:
                if self._current_system:
                    self._display_system(self._current_system)
                else:
                    print("  (no system prompt)")
                continue
            if stripped == "/done" and brace_depth == 0:
                return ["/done"]

            # Track brace depth for multi-line JSON
            brace_depth += stripped.count("{") - stripped.count("}")
            brace_depth = max(0, brace_depth)

            # Blank line = submit (only if we have content and braces balanced)
            if stripped == "" and lines and brace_depth == 0:
                break

            # Skip leading blank lines
            if stripped == "" and not lines:
                continue

            lines.append(line)

        return lines

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _merge_multiline_tools(self, lines: list[str]) -> list[str]:
        """Merge multi-line /tool JSON blocks into single lines."""
        merged: list[str] = []
        accumulator: list[str] | None = None
        acc_prefix: str = ""

        for line in lines:
            if accumulator is not None:
                accumulator.append(line)
                json_str = "\n".join(accumulator)
                try:
                    json.loads(json_str)
                    merged.append(f"{acc_prefix}{json_str}")
                    accumulator = None
                except json.JSONDecodeError:
                    continue
            elif line.strip().startswith("/tool "):
                rest = line.strip()[6:].strip()
                brace_idx = rest.find("{")
                if brace_idx != -1:
                    name = rest[:brace_idx].strip()
                    json_part = rest[brace_idx:]
                    try:
                        json.loads(json_part)
                        merged.append(line)
                    except json.JSONDecodeError:
                        acc_prefix = f"/tool {name} "
                        accumulator = [json_part]
                else:
                    merged.append(line)
            else:
                merged.append(line)

        # Unclosed JSON: emit as-is (will fail parse and become text with warning)
        if accumulator is not None:
            merged.append(f"{acc_prefix}{''.join(accumulator)}")

        return merged

    def _parse_tool_call(self, line: str) -> ToolCall | None:
        """Parse a single /tool line into a ToolCall."""
        rest = line.strip()[6:].strip()  # strip "/tool "

        brace_idx = rest.find("{")
        if brace_idx == -1:
            name = rest.strip()
            args: dict[str, Any] = {}
        else:
            name = rest[:brace_idx].strip()
            json_str = rest[brace_idx:]
            try:
                args = json.loads(json_str)
            except json.JSONDecodeError as e:
                c = self._c
                print(f"{c.RED}JSON parse error: {e}{c.RESET}")
                return None

        if not name:
            return None

        self._tc_counter += 1
        tc_id = f"human_tc_{self._tc_counter:03d}"
        return ToolCall(id=tc_id, name=name, arguments=args)

    def _parse_response(self, lines: list[str]) -> LLMResponse:
        """Parse collected lines into an LLMResponse."""
        if not lines or lines == ["/done"]:
            return LLMResponse(
                content=None,
                tool_calls=[],
                usage={"input_tokens": 0, "output_tokens": 0},
                stop_reason="end_turn",
            )

        merged = self._merge_multiline_tools(lines)

        text_lines: list[str] = []
        tool_calls: list[ToolCall] = []

        for line in merged:
            if line.strip().startswith("/tool "):
                tc = self._parse_tool_call(line)
                if tc:
                    tool_calls.append(tc)
                else:
                    c = self._c
                    print(
                        f"{c.RED}Warning: could not parse tool call, "
                        f"treating as text{c.RESET}"
                    )
                    text_lines.append(line)
            else:
                text_lines.append(line)

        content = "\n".join(text_lines).strip() or None

        if tool_calls:
            stop_reason = "tool_use"
        else:
            stop_reason = "stop"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={"input_tokens": 0, "output_tokens": 0},
            stop_reason=stop_reason,
        )
