"""Async bridge for consuming sync manager from async contexts.

This module provides async wrappers for the synchronous manager functions,
allowing the async FastAPI dashboard to call them without blocking the
event loop.

Architecture:
- Quick operations (start, stop, status): Run in ThreadPoolExecutor, await result
- Long operations (load_model): Submit to TaskManager, return Task immediately

The bridge uses a shared ThreadPoolExecutor with 4 workers:
- 1 for potential long-running operation
- 2-3 for concurrent quick operations (status, health checks)
"""

import asyncio
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, ParamSpec, TypeVar

from .control import (
    start_backend,
    stop_backend,
    stop_all,
    status,
    list_models,
    download_model,
    load_model,
    unload_model,
    get_backend_state,
    get_loaded_model,
)
from .tasks import Task, get_task_manager

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# Shared executor for all async bridge operations
# 4 workers: 1 long op + 2-3 concurrent quick ops
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bridge-")


async def run_sync(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a synchronous function in the thread pool.

    This is the core bridge function that allows async code to call
    sync functions without blocking the event loop.

    Args:
        func: Synchronous function to call
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        The return value of func
    """
    logger.debug(f"Bridge: run_sync({func.__name__}) called")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: func(*args, **kwargs)
    )
    logger.debug(f"Bridge: run_sync({func.__name__}) completed")
    return result


# ─────────────────────────────────────────────────────────────────
# Quick operations (run in executor, await result)
# ─────────────────────────────────────────────────────────────────

async def async_start_backend(name: str, force: bool = True) -> tuple[bool, str]:
    """Start a backend (async wrapper).

    Args:
        name: Backend name (e.g., "llama", "foundry")
        force: If True, stop any currently running backend first

    Returns:
        (success, message) tuple
    """
    logger.info(f"Bridge: Starting backend {name}")
    return await run_sync(start_backend, name, force=force)


async def async_stop_backend(backend) -> bool:
    """Stop a backend (async wrapper).

    Args:
        backend: Backend instance to stop

    Returns:
        True if stopped successfully
    """
    logger.info(f"Bridge: Stopping backend {backend.name if hasattr(backend, 'name') else backend}")
    return await run_sync(stop_backend, backend)


async def async_stop_all() -> bool:
    """Stop all backends (async wrapper).

    Returns:
        True if all backends stopped successfully
    """
    logger.info("Bridge: Stopping all backends")
    return await run_sync(stop_all)


async def async_status():
    """Get status of all backends (async wrapper).

    Returns:
        Status object with active backend info
    """
    logger.debug("Bridge: Getting status")
    return await run_sync(status)


async def async_list_models(backend_name: str | None = None) -> dict:
    """List models for a backend (async wrapper).

    Args:
        backend_name: Backend name, or None for active backend

    Returns:
        Dict mapping backend name to list of models
    """
    logger.debug(f"Bridge: Listing models for {backend_name or 'active backend'}")
    return await run_sync(list_models, backend_name)


async def async_get_backend_state(backend_name: str):
    """Get detailed state of a backend (async wrapper).

    Args:
        backend_name: Backend name

    Returns:
        BackendState object or None
    """
    logger.debug(f"Bridge: Getting backend state for {backend_name}")
    return await run_sync(get_backend_state, backend_name)


async def async_unload_model(backend_name: str) -> tuple[bool, str]:
    """Unload the current model (async wrapper).

    Args:
        backend_name: Backend to unload from

    Returns:
        (success, message) tuple
    """
    logger.info(f"Bridge: Unloading model from {backend_name}")
    return await run_sync(unload_model, backend_name)


async def async_get_loaded_model(backend_name: str):
    """Get the currently loaded model (async wrapper).

    Args:
        backend_name: Backend name

    Returns:
        Model object or None
    """
    logger.debug(f"Bridge: Getting loaded model for {backend_name}")
    return await run_sync(get_loaded_model, backend_name)


# ─────────────────────────────────────────────────────────────────
# Long operations (submit to TaskManager, return Task immediately)
# ─────────────────────────────────────────────────────────────────

async def async_load_model(backend_name: str, model_id: str) -> Task:
    """Load a model (returns Task, progress via callbacks).

    This submits the model loading to the TaskManager and returns
    immediately. The actual loading happens in a background thread.
    Use TaskManager callbacks or poll the Task for progress.

    Args:
        backend_name: Backend to load on
        model_id: Model identifier to load

    Returns:
        Task object for tracking progress
    """
    logger.info(f"Bridge: Submitting load_model task for {model_id} on {backend_name}")

    task_manager = get_task_manager()

    # Check if an operation is already in progress
    if task_manager.is_operation_in_progress():
        logger.warning("Bridge: Operation already in progress, cannot load model")
        # Return a failed task
        task = Task(
            id="rejected",
            operation="load_model",
            backend=backend_name,
            params={"backend_name": backend_name, "model_id": model_id},
        )
        task.status = task.status.FAILED
        task.error = "Another operation is already in progress"
        return task

    # Submit to task manager
    task = task_manager.submit(
        operation="load_model",
        backend=backend_name,
        func=load_model,
        params={"backend_name": backend_name, "model_id": model_id},
    )

    return task


async def async_download_model(backend_name: str, model_id: str) -> Task:
    """Download a model (returns Task, progress via callbacks).

    This submits the model download to the TaskManager and returns
    immediately. The actual download happens in a background thread.

    Args:
        backend_name: Backend to download for
        model_id: Model identifier to download

    Returns:
        Task object for tracking progress
    """
    logger.info(f"Bridge: Submitting download_model task for {model_id} on {backend_name}")

    task_manager = get_task_manager()

    # Check if an operation is already in progress
    if task_manager.is_operation_in_progress():
        logger.warning("Bridge: Operation already in progress, cannot download model")
        task = Task(
            id="rejected",
            operation="download_model",
            backend=backend_name,
            params={"backend_name": backend_name, "model_id": model_id},
        )
        task.status = task.status.FAILED
        task.error = "Another operation is already in progress"
        return task

    # Submit to task manager
    task = task_manager.submit(
        operation="download_model",
        backend=backend_name,
        func=download_model,
        params={"backend_name": backend_name, "model_id": model_id},
    )

    return task


def shutdown():
    """Shutdown the bridge executor and stop all backends.

    Call this when the application is shutting down.
    Stops backends first, then shuts down executors without waiting.
    """
    logger.info("Bridge: Shutting down")

    # Stop all running backends first
    try:
        logger.debug("Bridge: Stopping all backends")
        stop_all()
    except Exception as e:
        logger.warning(f"Bridge: Error stopping backends: {e}")

    # Kill any orphaned llama-server processes that might have escaped cleanup
    # This is a safety net - processes should normally be stopped by stop_all()
    try:
        logger.debug("Bridge: Killing any orphaned llama-server processes")
        subprocess.run(["pkill", "-15", "llama-server"], capture_output=True, timeout=5)
    except Exception:
        pass  # Ignore errors - process may not exist

    # Shutdown executors - don't wait, just cancel pending work
    logger.debug("Bridge: Shutting down task manager")
    get_task_manager().shutdown(wait=False, cancel_futures=True)

    logger.debug("Bridge: Shutting down executor")
    _executor.shutdown(wait=False, cancel_futures=True)

    logger.info("Bridge: Shutdown complete")
