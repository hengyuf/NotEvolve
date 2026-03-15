"""Kernel lifecycle management wrapping jupyter_client."""

from __future__ import annotations

import asyncio
import logging

from jupyter_client import AsyncKernelManager as JupyterAsyncKernelManager
from jupyter_client.asynchronous.client import AsyncKernelClient

logger = logging.getLogger(__name__)


class KernelManager:
    """Manages a single Jupyter kernel process with persistent state."""

    def __init__(self, kernel_name: str = "python3"):
        self._kernel_name = kernel_name
        self._km: JupyterAsyncKernelManager | None = None
        self._kc: AsyncKernelClient | None = None

    async def start(self) -> None:
        """Start the kernel process and connect client."""
        if self._km is not None:
            logger.warning("Kernel already started, restarting...")
            await self.shutdown()

        self._km = JupyterAsyncKernelManager(kernel_name=self._kernel_name)
        await self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()

        # Wait for kernel to be ready
        try:
            await asyncio.wait_for(self._kc.wait_for_ready(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("Kernel did not become ready within 30 seconds")
            await self.shutdown()
            raise RuntimeError("Kernel failed to start")

        logger.info("Kernel started: %s", self._kernel_name)

    async def shutdown(self) -> None:
        """Gracefully shut down the kernel."""
        if self._kc is not None:
            self._kc.stop_channels()
            self._kc = None

        if self._km is not None:
            await self._km.shutdown_kernel(now=True)
            self._km = None

        logger.info("Kernel shut down")

    async def restart(self) -> None:
        """Restart the kernel (clears all execution state)."""
        if self._km is None:
            await self.start()
            return

        if self._kc is not None:
            self._kc.stop_channels()

        await self._km.restart_kernel(now=True)

        self._kc = self._km.client()
        self._kc.start_channels()

        try:
            await asyncio.wait_for(self._kc.wait_for_ready(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("Kernel did not become ready after restart")
            raise RuntimeError("Kernel restart failed")

        logger.info("Kernel restarted")

    async def interrupt(self) -> None:
        """Interrupt the current execution."""
        if self._km is not None:
            await self._km.interrupt_kernel()
            logger.info("Kernel interrupted")

    @property
    def is_alive(self) -> bool:
        """Check if the kernel process is running.

        Uses the synchronous has_kernel check since the async is_alive()
        requires awaiting. For a more thorough check, use is_alive_async().
        """
        if self._km is None:
            return False
        return self._km.has_kernel

    async def is_alive_async(self) -> bool:
        """Async check if the kernel process is running."""
        if self._km is None:
            return False
        return await self._km.is_alive()

    @property
    def client(self) -> AsyncKernelClient:
        """Get the kernel client. Raises if not started."""
        if self._kc is None:
            raise RuntimeError("Kernel not started. Call start() first.")
        return self._kc
