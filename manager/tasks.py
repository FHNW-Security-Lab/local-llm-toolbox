"""Task management for long-running operations.

This module provides a task system for managing long-running backend operations
(like model loading) without blocking the main thread. It uses a global lock
to ensure only one operation runs at a time (since only one backend/model can
be active).

Key components:
- TaskStatus: Enum for task lifecycle states
- Task: Dataclass representing a long-running operation
- TaskManager: Manages task submission, execution, and callbacks
"""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task lifecycle status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A long-running operation.

    Attributes:
        id: Unique task identifier
        operation: Type of operation (e.g., "load_model", "download_model")
        backend: Backend this task operates on
        params: Parameters passed to the operation function
        status: Current task status
        progress: Progress from 0.0 to 1.0
        message: Human-readable status message
        error: Error message if failed
        result: Return value from the operation
        started_at: Unix timestamp when task started running
        completed_at: Unix timestamp when task completed/failed
    """
    id: str
    operation: str
    backend: str
    params: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    result: Any = None
    started_at: float | None = None
    completed_at: float | None = None


class TaskManager:
    """Manages long-running backend operations.

    Uses a global lock to ensure only one operation runs at a time,
    since only one backend/model can be active. Operations are executed
    in a thread pool to avoid blocking the caller.

    Example:
        manager = get_task_manager()

        # Submit a task
        task = manager.submit(
            operation="load_model",
            backend="llama",
            func=backend.load_model,
            params={"model_id": "my-model"},
        )

        # Check status
        task = manager.get_task(task.id)
        print(task.status, task.message)

        # Register for updates
        manager.on_task_update(lambda t: print(f"Task {t.id}: {t.status}"))
    """

    def __init__(self, max_workers: int = 2):
        """Initialize the task manager.

        Args:
            max_workers: Maximum concurrent tasks in the executor.
                         Default is 2 (one long op + one quick check).
        """
        self._tasks: dict[str, Task] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="task-",
        )
        # Global lock - only one operation at a time
        self._operation_lock = threading.Lock()
        self._callbacks: list[Callable[[Task], None]] = []
        self._lock = threading.Lock()  # Protects _tasks and _callbacks

        logger.debug(f"TaskManager initialized with {max_workers} workers")

    def submit(
        self,
        operation: str,
        backend: str,
        func: Callable[..., tuple[bool, str]],
        params: dict | None = None,
    ) -> Task:
        """Submit a long-running operation.

        Args:
            operation: Name of the operation (e.g., "load_model")
            backend: Backend name this operates on
            func: Function to execute, must return (success: bool, message: str)
            params: Keyword arguments to pass to func

        Returns:
            Task object that can be used to track progress
        """
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            id=task_id,
            operation=operation,
            backend=backend,
            params=params or {},
        )

        with self._lock:
            self._tasks[task_id] = task

        logger.info(f"Task {task_id} submitted: {operation} on {backend}")
        logger.debug(f"Task {task_id} params: {params}")

        # Submit to executor
        self._executor.submit(self._run_task, task, func)

        return task

    def _run_task(self, task: Task, func: Callable):
        """Execute a task with global locking.

        This runs in a worker thread. It acquires the global operation lock
        to ensure only one operation runs at a time.
        """
        logger.debug(f"Task {task.id}: Attempting to acquire operation lock")

        # Try to acquire lock without blocking
        if not self._operation_lock.acquire(blocking=False):
            logger.warning(f"Task {task.id}: Operation lock busy, failing task")
            task.status = TaskStatus.FAILED
            task.error = "Another operation is already in progress"
            task.completed_at = time.time()
            self._notify(task)
            return

        logger.debug(f"Task {task.id}: Lock acquired, thread={threading.current_thread().name}")

        try:
            # Transition to running
            old_status = task.status
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.message = f"Starting {task.operation}..."
            logger.debug(f"Task {task.id}: State transition {old_status} -> {task.status}")
            self._notify(task)

            # Execute the operation
            logger.info(f"Task {task.id}: Executing {task.operation}")
            success, message = func(**task.params)

            # Record completion
            task.completed_at = time.time()
            elapsed = task.completed_at - task.started_at

            if success:
                old_status = task.status
                task.status = TaskStatus.COMPLETED
                task.message = message
                task.progress = 1.0
                task.result = (success, message)
                logger.info(f"Task {task.id}: Completed in {elapsed:.1f}s - {message}")
            else:
                old_status = task.status
                task.status = TaskStatus.FAILED
                task.error = message
                task.result = (success, message)
                logger.warning(f"Task {task.id}: Failed after {elapsed:.1f}s - {message}")

            logger.debug(f"Task {task.id}: State transition {old_status} -> {task.status}")
            self._notify(task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            logger.exception(f"Task {task.id}: Exception during execution")
            self._notify(task)

        finally:
            logger.debug(f"Task {task.id}: Releasing operation lock")
            self._operation_lock.release()

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: The task identifier

        Returns:
            Task object or None if not found
        """
        with self._lock:
            return self._tasks.get(task_id)

    def get_active_task(self) -> Task | None:
        """Get the currently running task (if any).

        Returns:
            The running task, or None if no task is running
        """
        with self._lock:
            for task in self._tasks.values():
                if task.status == TaskStatus.RUNNING:
                    return task
        return None

    def get_recent_tasks(self, limit: int = 10) -> list[Task]:
        """Get recent tasks, newest first.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of tasks, sorted by start time (newest first)
        """
        with self._lock:
            tasks = list(self._tasks.values())

        # Sort by started_at (or 0 if not started), newest first
        tasks.sort(key=lambda t: t.started_at or 0, reverse=True)
        return tasks[:limit]

    def is_operation_in_progress(self) -> bool:
        """Check if any operation is currently running.

        Returns:
            True if an operation is in progress
        """
        return self._operation_lock.locked()

    def on_task_update(self, callback: Callable[[Task], None]):
        """Register a callback for task updates.

        The callback will be called whenever a task changes state
        (started, completed, failed).

        Args:
            callback: Function that takes a Task as argument
        """
        with self._lock:
            self._callbacks.append(callback)
        logger.debug(f"Registered task update callback, total={len(self._callbacks)}")

    def _notify(self, task: Task):
        """Notify all callbacks of a task update."""
        with self._lock:
            callbacks = list(self._callbacks)

        for cb in callbacks:
            try:
                cb(task)
            except Exception as e:
                logger.warning(f"Task callback error: {e}")

    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        """Shutdown the task manager.

        Args:
            wait: If True, wait for pending tasks to complete
            cancel_futures: If True, cancel pending futures (requires Python 3.9+)
        """
        logger.info(f"TaskManager shutting down (wait={wait}, cancel={cancel_futures})")
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)


# Global task manager instance
_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """Get the global task manager.

    Returns:
        The singleton TaskManager instance
    """
    global _manager
    if _manager is None:
        _manager = TaskManager()
    return _manager
