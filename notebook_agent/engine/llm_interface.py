"""LLM API adapter supporting Anthropic and OpenAI-compatible APIs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None


class LLMInterface:
    """Adapter for LLM API calls."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        max_tokens: int = 16384,
        thinking_budget: int = 10000,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget
        self._base_url = base_url
        self._extra_headers = extra_headers or {}
        self._client: Any = None

    def _ensure_client(self) -> None:
        """Lazily initialize the API client."""
        if self._client is not None:
            return

        provider = self._provider.lower()
        if provider == "anthropic":
            import anthropic

            kwargs: dict[str, Any] = {"api_key": self._api_key or None}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._extra_headers:
                kwargs["default_headers"] = self._extra_headers
            self._client = anthropic.Anthropic(**kwargs)
            return

        if provider in ("openai", "openai_compatible"):
            import openai

            kwargs = {"api_key": self._api_key or None}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._extra_headers:
                kwargs["default_headers"] = self._extra_headers
            self._client = openai.OpenAI(**kwargs)
            return

        raise ValueError(f"Unsupported provider: {self._provider}")

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 1.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and return response."""
        self._ensure_client()

        provider = self._provider.lower()
        if provider == "anthropic":
            return await self._chat_anthropic(messages, tools, temperature, system)
        if provider in ("openai", "openai_compatible"):
            return await self._chat_openai(messages, tools, temperature, system)
        raise ValueError(f"Unsupported provider: {self._provider}")

    async def _chat_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        system: str | None,
    ) -> LLMResponse:
        """Call Anthropic Messages API."""
        import anthropic

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if self._thinking_budget > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}
            kwargs["temperature"] = 1.0

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APIError as e:
            logger.error("Anthropic API error: %s", e)
            raise

        content_text = None
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage={
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            },
            stop_reason=response.stop_reason,
        )

    def _convert_messages_to_openai(
        self, messages: list[dict], system: str | None
    ) -> list[dict]:
        """Convert orchestrator's Anthropic-style message blocks to OpenAI format."""
        converted: list[dict] = []
        if system:
            converted.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            if role == "assistant" and isinstance(content, list):
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                        )

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(t for t in text_parts if t) if text_parts else "",
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                converted.append(assistant_msg)
                continue

            if role == "user" and isinstance(content, list):
                # Anthropic-style tool results are represented as user blocks.
                if all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
                    for block in content:
                        converted.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id"),
                                "content": str(block.get("content", "")),
                            }
                        )
                    continue

                converted.append({"role": "user", "content": json.dumps(content)})
                continue

            converted.append({"role": role or "user", "content": str(content)})

        return converted

    async def _chat_openai(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        system: str | None,
    ) -> LLMResponse:
        """Call OpenAI Chat Completions API (also used for OpenAI-compatible APIs)."""
        converted_messages = self._convert_messages_to_openai(messages, system)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
                for tool in tools
            ]

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content_text = choice.message.content
        tool_calls: list[ToolCall] = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                arguments_raw = tc.function.arguments or "{}"
                try:
                    parsed_arguments = json.loads(arguments_raw)
                    if not isinstance(parsed_arguments, dict):
                        parsed_arguments = {"value": parsed_arguments}
                except json.JSONDecodeError:
                    parsed_arguments = {"raw": arguments_raw}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=parsed_arguments,
                    )
                )

        stop_reason_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
        }
        stop_reason = stop_reason_map.get(choice.finish_reason, choice.finish_reason)

        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            },
            stop_reason=stop_reason,
        )

