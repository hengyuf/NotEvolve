"""Tool registry for dispatching tool calls."""

from __future__ import annotations

import logging
from typing import Any

from notebook_agent.security.policy import SecurityError, SecurityPolicy
from notebook_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry mapping tool names to handlers with security checks."""

    def __init__(self, security: SecurityPolicy | None = None):
        self._tools: dict[str, BaseTool] = {}
        self._security = security

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_all(self, tools: list[BaseTool]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def get_schemas(self) -> list[dict]:
        """Return JSON schemas for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool call with security checks."""
        if tool_name not in self._tools:
            return ToolResult(
                content=f"Error: unknown tool '{tool_name}'. Available tools: {', '.join(self._tools.keys())}",
                is_error=True,
            )

        # Security check
        if self._security:
            try:
                self._security.check_tool_call(tool_name, arguments)
            except SecurityError as e:
                logger.warning("Security blocked tool call %s: %s", tool_name, e)
                return ToolResult(content=f"Security error: {e}", is_error=True)

        tool = self._tools[tool_name]
        try:
            result = await tool.execute(**arguments)
            return result
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
            return ToolResult(content=f"Error executing {tool_name}: {e}", is_error=True)
