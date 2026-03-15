"""Tests for kernel manager and executor.

These tests require a working Jupyter kernel (ipykernel).
They are integration tests and may be slower.
"""

import pytest

from notebook_agent.kernel.executor import CellExecutor
from notebook_agent.kernel.manager import KernelManager


@pytest.fixture
async def kernel():
    """Start and yield a kernel, shut down after test."""
    km = KernelManager(kernel_name="python3")
    await km.start()
    yield km
    await km.shutdown()


@pytest.mark.asyncio
class TestKernelManager:
    async def test_start_and_alive(self, kernel):
        assert kernel.is_alive

    async def test_restart(self, kernel):
        await kernel.restart()
        assert kernel.is_alive

    async def test_shutdown(self):
        km = KernelManager()
        await km.start()
        assert km.is_alive
        await km.shutdown()
        assert not km.is_alive


@pytest.mark.asyncio
class TestCellExecutor:
    async def test_simple_execution(self, kernel):
        executor = CellExecutor(kernel)
        result = await executor.execute("1 + 1")
        assert result.status == "ok"
        assert result.execution_count is not None

    async def test_print_output(self, kernel):
        executor = CellExecutor(kernel)
        result = await executor.execute("print('hello world')")
        assert result.status == "ok"
        assert any("hello world" in out.content for out in result.outputs)

    async def test_error_execution(self, kernel):
        executor = CellExecutor(kernel)
        result = await executor.execute("1 / 0")
        assert result.status == "error"
        assert result.error is not None
        assert "ZeroDivision" in result.error

    async def test_variable_persistence(self, kernel):
        executor = CellExecutor(kernel)
        await executor.execute("my_var = 42")
        result = await executor.execute("print(my_var)")
        assert result.status == "ok"
        assert any("42" in out.content for out in result.outputs)

    async def test_multiline_output(self, kernel):
        executor = CellExecutor(kernel)
        code = "for i in range(5):\n    print(f'line {i}')"
        result = await executor.execute(code)
        assert result.status == "ok"
        output_text = "\n".join(out.content for out in result.outputs)
        assert "line 0" in output_text
        assert "line 4" in output_text

    async def test_dead_kernel(self):
        km = KernelManager()
        executor = CellExecutor(km)
        result = await executor.execute("x = 1")
        assert result.status == "error"
        assert "not running" in result.error
