"""Base tool abstraction and ToolResult."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result returned from tool execution."""

    content: str
    is_error: bool = False


class BaseTool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in LLM tool calls."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for tool parameters."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""

    def get_schema(self) -> dict:
        """Return the full tool schema for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
