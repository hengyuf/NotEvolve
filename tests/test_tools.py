"""Tests for tool registry and basic tool functionality."""

import pytest

from notebook_agent.security.policy import SecurityPolicy
from notebook_agent.tools.base import BaseTool, ToolResult
from notebook_agent.tools.evaluator_tools import load_evaluator_tools
from notebook_agent.tools.file_tools import GlobFilesTool, GrepFilesTool
from notebook_agent.tools.registry import ToolRegistry


class DummyTool(BaseTool):
    name = "dummy"
    description = "A test tool."
    parameters = {"type": "object", "properties": {"msg": {"type": "string"}}}

    async def execute(self, msg: str = "hello", **kwargs) -> ToolResult:
        return ToolResult(content=f"Got: {msg}")


class FailingTool(BaseTool):
    name = "fail"
    description = "A tool that always fails."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("intentional failure")


@pytest.mark.asyncio
class TestToolRegistry:
    async def test_register_and_dispatch(self):
        registry = ToolRegistry()
        registry.register(DummyTool())

        result = await registry.dispatch("dummy", {"msg": "world"})
        assert result.content == "Got: world"
        assert not result.is_error

    async def test_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.dispatch("nonexistent", {})
        assert result.is_error
        assert "unknown tool" in result.content

    async def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register(DummyTool())
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "dummy"
        assert "input_schema" in schemas[0]

    async def test_tool_failure_returns_error(self):
        registry = ToolRegistry()
        registry.register(FailingTool())

        result = await registry.dispatch("fail", {})
        assert result.is_error
        assert "intentional failure" in result.content

    async def test_security_blocks_tool(self, tmp_path):
        eval_dir = tmp_path / "evaluator"
        eval_dir.mkdir()
        secret = eval_dir / "secret.py"
        secret.write_text("")

        security = SecurityPolicy(evaluator_paths=[str(eval_dir)])
        registry = ToolRegistry(security=security)
        registry.register(DummyTool())

        # Tool call with protected path
        result = await registry.dispatch("dummy", {"path": str(secret)})
        assert result.is_error
        assert "Security" in result.content

    async def test_load_evaluator_tools_from_file(self, tmp_path):
        adapter = tmp_path / "adapter.py"
        adapter.write_text(
            "\n".join(
                [
                    "def evaluate_distribution(support: list[float], probs: list[float]) -> str:",
                    "    return f'len={len(support)}'",
                    "",
                    "def check_score() -> str:",
                    "    return 'ok'",
                ]
            ),
            encoding="utf-8",
        )
        tools = load_evaluator_tools({"module": str(adapter)})
        names = sorted(t.name for t in tools)
        assert "check_score" in names
        assert "evaluate_distribution" in names

    async def test_glob_files_resolves_relative_path_from_working_dir(self, tmp_path):
        workdir = tmp_path / "workspace"
        target_dir = workdir / "src"
        target_dir.mkdir(parents=True)
        (target_dir / "main.py").write_text("print('hi')", encoding="utf-8")

        security = SecurityPolicy(working_dir=str(workdir))
        tool = GlobFilesTool(str(workdir), security)

        result = await tool.execute(pattern="*.py", path="src")
        assert not result.is_error
        assert "src/main.py" in result.content

    async def test_grep_files_resolves_relative_path_from_working_dir(self, tmp_path):
        workdir = tmp_path / "workspace"
        target_dir = workdir / "src"
        target_dir.mkdir(parents=True)
        (target_dir / "main.py").write_text("needle = 1\n", encoding="utf-8")

        security = SecurityPolicy(working_dir=str(workdir))
        tool = GrepFilesTool(str(workdir), security)

        result = await tool.execute(pattern="needle", path="src")
        assert not result.is_error
        assert "src/main.py:1:needle = 1" in result.content
