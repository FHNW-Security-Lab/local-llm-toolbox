"""Generic process management utilities."""

import asyncio
import logging
import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Information about a managed process."""
    name: str
    pid: int | None = None
    cmd: list[str] = field(default_factory=list)
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    process: subprocess.Popen | None = field(default=None, repr=False)


class ProcessManager:
    """Manages spawning and stopping of processes."""

    def __init__(self, graceful_timeout: int = 15):
        self.graceful_timeout = graceful_timeout
        self._processes: dict[str, ProcessInfo] = {}

    def start(
        self,
        name: str,
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        stdout: int | None = subprocess.DEVNULL,
        stderr: int | None = subprocess.DEVNULL,
    ) -> tuple[bool, str]:
        """
        Start a process.

        Returns (success, message).
        """
        if name in self._processes and self._processes[name].process:
            proc = self._processes[name].process
            if proc.poll() is None:
                return False, f"Process '{name}' is already running (PID: {proc.pid})"

        # Merge environment
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            logger.info(f"Starting process '{name}': {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=full_env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,  # Detach from parent
            )

            self._processes[name] = ProcessInfo(
                name=name,
                pid=process.pid,
                cmd=cmd,
                cwd=cwd,
                env=env or {},
                process=process,
            )

            logger.info(f"Started '{name}' with PID {process.pid}")
            return True, f"Started {name} (PID: {process.pid})"

        except Exception as e:
            logger.exception(f"Failed to start '{name}'")
            return False, f"Failed to start {name}: {e}"

    def stop(self, name: str, force: bool = False) -> tuple[bool, str]:
        """
        Stop a process.

        Args:
            name: Process name
            force: If True, use SIGKILL immediately. Otherwise try SIGTERM first.

        Returns (success, message).
        """
        if name not in self._processes:
            return False, f"No process named '{name}'"

        info = self._processes[name]
        if not info.process or info.process.poll() is not None:
            del self._processes[name]
            return True, f"Process '{name}' was not running"

        proc = info.process
        pid = proc.pid

        try:
            if force:
                logger.info(f"Force killing '{name}' (PID: {pid})")
                proc.kill()
                proc.wait(timeout=5)
            else:
                logger.info(f"Gracefully stopping '{name}' (PID: {pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=self.graceful_timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Graceful shutdown timed out, killing '{name}'")
                    proc.kill()
                    proc.wait(timeout=5)

            del self._processes[name]
            logger.info(f"Stopped '{name}' (PID: {pid})")
            return True, f"Stopped {name} (PID: {pid})"

        except Exception as e:
            logger.exception(f"Failed to stop '{name}'")
            return False, f"Failed to stop {name}: {e}"

    def stop_all(self, force: bool = False) -> tuple[bool, str]:
        """Stop all managed processes."""
        names = list(self._processes.keys())
        if not names:
            return True, "No processes to stop"

        errors = []
        for name in names:
            success, msg = self.stop(name, force=force)
            if not success:
                errors.append(msg)

        if errors:
            return False, "; ".join(errors)
        return True, f"Stopped {len(names)} process(es)"

    def is_running(self, name: str) -> bool:
        """Check if a process is running."""
        if name not in self._processes:
            return False
        proc = self._processes[name].process
        return proc is not None and proc.poll() is None

    def get_pid(self, name: str) -> int | None:
        """Get PID of a running process."""
        if name in self._processes and self.is_running(name):
            return self._processes[name].pid
        return None

    def list_processes(self) -> list[ProcessInfo]:
        """List all managed processes."""
        return list(self._processes.values())


async def wait_for_healthy(
    url: str,
    timeout: float = 60.0,
    interval: float = 0.5,
    expected_status: int = 200,
) -> bool:
    """
    Wait for an HTTP endpoint to become healthy.

    Args:
        url: Health check URL
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds
        expected_status: Expected HTTP status code (or any < 500 if None)

    Returns True if healthy within timeout, False otherwise.
    """
    logger.debug(f"Waiting for {url} to become healthy (timeout: {timeout}s)")
    deadline = asyncio.get_event_loop().time() + timeout

    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code < 500:
                    logger.debug(f"{url} is healthy (status: {resp.status_code})")
                    return True
            except httpx.RequestError:
                pass

            await asyncio.sleep(interval)

    logger.warning(f"{url} did not become healthy within {timeout}s")
    return False


def check_health_sync(url: str, timeout: float = 2.0) -> bool:
    """Synchronous health check."""
    try:
        resp = httpx.get(url, timeout=timeout)
        return resp.status_code < 500
    except httpx.RequestError:
        return False


def kill_process_by_pid(pid: int, force: bool = False) -> bool:
    """Kill a process by PID."""
    try:
        sig = signal.SIGKILL if force else signal.SIGTERM
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return True  # Already dead
    except Exception:
        return False


def find_process_by_port(port: int) -> int | None:
    """Find PID of process listening on a port."""
    try:
        # Try lsof (macOS/Linux)
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split()[0])
    except Exception:
        pass
    return None
