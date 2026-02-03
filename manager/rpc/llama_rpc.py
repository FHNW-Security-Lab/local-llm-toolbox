"""llama.cpp RPC worker management.

This module provides:
- RpcWorkerServer: Runs on worker nodes, manages rpc-server + control API
- RpcWorkerClient: Used by main node to communicate with workers

The control API allows the main node to reset workers before loading a new model,
ensuring a clean state without orphaned GPU memory allocations.
"""

import asyncio
import logging
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Default ports
DEFAULT_RPC_PORT = 50052
DEFAULT_CONTROL_PORT = 50053

# Cache directory for llama.cpp RPC tensor cache
RPC_CACHE_DIR = Path.home() / ".cache" / "llama.cpp" / "rpc"


# =============================================================================
# API Models
# =============================================================================


class HealthResponse(BaseModel):
    status: str = "ok"


class StatusResponse(BaseModel):
    rpc_port: int
    running: bool
    pid: int | None = None
    uptime_seconds: float | None = None


class ResetResponse(BaseModel):
    status: str
    message: str


# =============================================================================
# Worker Server (runs on worker nodes)
# =============================================================================


class RpcWorkerServer:
    """
    Runs on worker nodes via './toolbox rpc llama'.

    Manages:
    - rpc-server process (llama.cpp's TCP-based tensor offload server)
    - Control API (HTTP endpoints for health, status, reset)

    Usage:
        server = RpcWorkerServer(rpc_port=50052, control_port=50053)
        server.run()  # Blocks until interrupted
    """

    def __init__(
        self,
        rpc_port: int = DEFAULT_RPC_PORT,
        control_port: int = DEFAULT_CONTROL_PORT,
    ):
        self.rpc_port = rpc_port
        self.control_port = control_port
        self._rpc_process: subprocess.Popen | None = None
        self._rpc_start_time: float | None = None

        # Create FastAPI app for control API
        self._app = FastAPI(title="llama.cpp RPC Worker Control API")
        self._setup_routes()

        logger.info(
            f"RpcWorkerServer initialized: rpc_port={rpc_port}, control_port={control_port}"
        )

    def _setup_routes(self) -> None:
        """Setup FastAPI routes for control API."""

        @self._app.get("/health", response_model=HealthResponse)
        async def health() -> HealthResponse:
            """Health check endpoint."""
            return HealthResponse(status="ok")

        @self._app.get("/status", response_model=StatusResponse)
        async def status() -> StatusResponse:
            """Get RPC server status."""
            running = self._is_rpc_running()
            pid = self._rpc_process.pid if self._rpc_process and running else None
            uptime = None
            if running and self._rpc_start_time:
                uptime = time.time() - self._rpc_start_time

            logger.debug(f"Status request: running={running}, pid={pid}, uptime={uptime}")
            return StatusResponse(
                rpc_port=self.rpc_port,
                running=running,
                pid=pid,
                uptime_seconds=uptime,
            )

        @self._app.post("/reset", response_model=ResetResponse)
        async def reset(
            force: bool = Query(False, description="Also clear tensor cache")
        ) -> ResetResponse:
            """
            Reset the RPC server.

            Stops any running rpc-server, optionally clears the tensor cache,
            and starts a fresh rpc-server instance.

            Args:
                force: If True, also clear the tensor cache directory
            """
            logger.info(f"Reset request received (force={force})")

            try:
                # Stop existing rpc-server
                if self._is_rpc_running():
                    logger.info("Stopping existing rpc-server...")
                    self._stop_rpc_server()
                    # Wait for process to fully terminate
                    await asyncio.sleep(1)

                # Clear cache if force reset
                if force:
                    logger.info(f"Clearing tensor cache at {RPC_CACHE_DIR}")
                    self._clear_cache()

                # Start fresh rpc-server
                logger.info("Starting fresh rpc-server...")
                if not self._start_rpc_server():
                    error_msg = "Failed to start rpc-server"
                    logger.error(error_msg)
                    return ResetResponse(status="error", message=error_msg)

                # Wait for it to be ready
                if not await self._wait_for_rpc_ready(timeout=10):
                    error_msg = "rpc-server started but port not reachable"
                    logger.error(error_msg)
                    return ResetResponse(status="error", message=error_msg)

                logger.info("Reset complete - rpc-server is ready")
                return ResetResponse(status="ok", message="RPC server reset successfully")

            except Exception as e:
                logger.exception(f"Reset failed: {e}")
                return ResetResponse(status="error", message=str(e))

    def _is_rpc_running(self) -> bool:
        """Check if rpc-server process is running."""
        if self._rpc_process is None:
            return False
        return self._rpc_process.poll() is None

    def _start_rpc_server(self) -> bool:
        """Start the rpc-server process."""
        # Check if rpc-server binary exists
        if shutil.which("rpc-server") is None:
            logger.error("rpc-server binary not found in PATH")
            return False

        cmd = [
            "rpc-server",
            "--host", "0.0.0.0",
            "--port", str(self.rpc_port),
            "-c",  # Enable tensor caching
        ]

        logger.info(f"Starting rpc-server: {' '.join(cmd)}")

        try:
            self._rpc_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent
            )
            self._rpc_start_time = time.time()
            logger.info(f"rpc-server started with PID {self._rpc_process.pid}")
            return True
        except Exception as e:
            logger.exception(f"Failed to start rpc-server: {e}")
            return False

    def _stop_rpc_server(self, force: bool = False) -> None:
        """Stop the rpc-server process."""
        if self._rpc_process is None:
            logger.debug("No rpc-server process to stop")
            return

        pid = self._rpc_process.pid
        logger.info(f"Stopping rpc-server (PID {pid}, force={force})")

        try:
            if force:
                self._rpc_process.kill()
            else:
                self._rpc_process.terminate()

            # Wait for process to exit (with timeout)
            try:
                self._rpc_process.wait(timeout=5)
                logger.info(f"rpc-server (PID {pid}) terminated")
            except subprocess.TimeoutExpired:
                logger.warning(f"rpc-server (PID {pid}) did not exit, force killing")
                self._rpc_process.kill()
                self._rpc_process.wait(timeout=5)

        except Exception as e:
            logger.warning(f"Error stopping rpc-server: {e}")

        self._rpc_process = None
        self._rpc_start_time = None

    def _clear_cache(self) -> None:
        """Clear the RPC tensor cache directory."""
        if RPC_CACHE_DIR.exists():
            try:
                shutil.rmtree(RPC_CACHE_DIR)
                logger.info(f"Cleared cache directory: {RPC_CACHE_DIR}")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
        else:
            logger.debug(f"Cache directory does not exist: {RPC_CACHE_DIR}")

    async def _wait_for_rpc_ready(self, timeout: float = 10) -> bool:
        """Wait for rpc-server port to become reachable."""
        deadline = time.time() + timeout
        check_count = 0

        while time.time() < deadline:
            check_count += 1
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", self.rpc_port))
                sock.close()
                if result == 0:
                    logger.debug(f"rpc-server ready after {check_count} checks")
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)

        logger.warning(f"Timeout waiting for rpc-server after {check_count} checks")
        return False

    def run(self) -> None:
        """Run the worker server (blocking)."""
        import uvicorn

        logger.info(f"Starting RPC worker server...")
        logger.info(f"  RPC server port:     {self.rpc_port}")
        logger.info(f"  Control API port:    {self.control_port}")

        # Start rpc-server on startup
        if not self._start_rpc_server():
            logger.error("Failed to start rpc-server on startup")
            return

        # Wait briefly for rpc-server to start
        time.sleep(1)
        if not self._is_rpc_running():
            logger.error("rpc-server died immediately after starting")
            return

        logger.info(f"rpc-server running on port {self.rpc_port}")
        logger.info(f"Control API starting on port {self.control_port}")

        # Setup signal handlers for graceful shutdown
        def shutdown_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._stop_rpc_server()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        try:
            # Run the control API (blocking)
            uvicorn.run(
                self._app,
                host="0.0.0.0",
                port=self.control_port,
                log_level="info",
            )
        finally:
            logger.info("Shutting down RPC worker server...")
            self._stop_rpc_server()


# =============================================================================
# Worker Client (used by main node)
# =============================================================================


class RpcWorkerClient:
    """
    Client for communicating with RPC workers from the main node.

    Used by LlamaBackend to:
    - Check worker health before loading models
    - Reset workers to ensure clean state
    - Get RPC port information for --rpc flag

    Usage:
        client = RpcWorkerClient(
            workers=["192.168.1.10", "192.168.1.11"],
            control_port=50053,
        )

        # Reset all workers before loading a model
        results = await client.reset_all_workers()

        # Get the --rpc flag value
        rpc_addresses = await client.get_rpc_addresses()
        # Returns: "192.168.1.10:50052,192.168.1.11:50052"
    """

    def __init__(
        self,
        workers: list[str],
        control_port: int = DEFAULT_CONTROL_PORT,
        timeout: float = 10.0,
    ):
        """
        Initialize the RPC worker client.

        Args:
            workers: List of worker hostnames/IPs
            control_port: Control API port (same for all workers)
            timeout: HTTP request timeout in seconds
        """
        self.workers = workers
        self.control_port = control_port
        self.timeout = timeout

        logger.info(
            f"RpcWorkerClient initialized: workers={workers}, control_port={control_port}"
        )

    async def health_check(self, host: str) -> bool:
        """
        Check if a worker's control API is reachable.

        Args:
            host: Worker hostname/IP

        Returns:
            True if worker responded to health check
        """
        url = f"http://{host}:{self.control_port}/health"
        logger.debug(f"Health check: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                healthy = response.status_code == 200
                logger.debug(f"Health check {host}: {healthy}")
                return healthy
        except Exception as e:
            logger.warning(f"Health check failed for {host}: {e}")
            return False

    async def get_status(self, host: str) -> StatusResponse | None:
        """
        Get status from a worker including its RPC port.

        Args:
            host: Worker hostname/IP

        Returns:
            StatusResponse if successful, None on error
        """
        url = f"http://{host}:{self.control_port}/status"
        logger.debug(f"Getting status: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    status = StatusResponse(**data)
                    logger.debug(f"Status from {host}: running={status.running}, rpc_port={status.rpc_port}")
                    return status
                else:
                    logger.warning(f"Status request to {host} failed: {response.status_code}")
                    return None
        except Exception as e:
            logger.warning(f"Failed to get status from {host}: {e}")
            return None

    async def reset_worker(self, host: str, force: bool = False) -> bool:
        """
        Reset a single worker.

        Args:
            host: Worker hostname/IP
            force: If True, also clear tensor cache

        Returns:
            True if reset was successful
        """
        url = f"http://{host}:{self.control_port}/reset"
        params = {"force": "true"} if force else {}
        logger.info(f"Resetting worker {host} (force={force})")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:  # Longer timeout for reset
                response = await client.post(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok":
                        logger.info(f"Worker {host} reset successfully")
                        return True
                    else:
                        logger.warning(f"Worker {host} reset failed: {data.get('message')}")
                        return False
                else:
                    logger.warning(f"Reset request to {host} failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.warning(f"Failed to reset worker {host}: {e}")
            return False

    async def reset_all_workers(self, force: bool = False) -> dict[str, bool]:
        """
        Reset all workers concurrently.

        If a worker fails, retries once before marking as failed.

        Args:
            force: If True, also clear tensor cache on all workers

        Returns:
            Dict mapping hostname to success status
        """
        logger.info(f"Resetting {len(self.workers)} workers (force={force})")

        async def reset_with_retry(host: str) -> tuple[str, bool]:
            """Reset a worker with one retry on failure."""
            success = await self.reset_worker(host, force=force)
            if not success:
                logger.info(f"Retrying reset for {host}...")
                await asyncio.sleep(1)
                success = await self.reset_worker(host, force=force)
            return host, success

        # Reset all workers concurrently
        tasks = [reset_with_retry(host) for host in self.workers]
        results = await asyncio.gather(*tasks)

        result_dict = dict(results)

        # Log summary
        success_count = sum(1 for v in result_dict.values() if v)
        logger.info(f"Reset complete: {success_count}/{len(self.workers)} workers successful")

        for host, success in result_dict.items():
            if not success:
                logger.warning(f"Worker {host} failed to reset - will be skipped")

        return result_dict

    async def get_rpc_addresses(self) -> str:
        """
        Get comma-separated RPC addresses for --rpc flag.

        Queries each worker for its RPC port and builds the address list.
        Only includes workers that respond successfully.

        Returns:
            Comma-separated list like "192.168.1.10:50052,192.168.1.11:50052"
        """
        logger.debug("Building RPC addresses list")

        async def get_address(host: str) -> str | None:
            status = await self.get_status(host)
            if status and status.running:
                return f"{host}:{status.rpc_port}"
            else:
                logger.warning(f"Worker {host} not running, excluding from RPC addresses")
                return None

        # Query all workers concurrently
        tasks = [get_address(host) for host in self.workers]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        addresses = [addr for addr in results if addr is not None]

        rpc_string = ",".join(addresses)
        logger.info(f"RPC addresses: {rpc_string}")
        return rpc_string

    def check_rpc_port_sync(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """
        Synchronously check if an RPC port is reachable (TCP connect).

        Args:
            host: Worker hostname/IP
            port: RPC port to check
            timeout: Connection timeout in seconds

        Returns:
            True if port is reachable
        """
        logger.debug(f"TCP check: {host}:{port}")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            reachable = result == 0
            logger.debug(f"TCP check {host}:{port}: {'reachable' if reachable else 'unreachable'}")
            return reachable
        except Exception as e:
            logger.debug(f"TCP check {host}:{port} failed: {e}")
            return False
