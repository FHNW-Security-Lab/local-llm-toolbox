"""Backend manager - shared logic for controlling LLM backends."""

from .interface import Backend, BackendState, Model, Node, ClusterConfig
from .registry import get_registry, get_backend, get_all_backends, get_active_backend
from .control import (
    start_backend,
    stop_backend,
    stop_all,
    status,
    list_models,
    download_model,
    load_model,
    unload_model,
    get_loaded_model,
    get_backend_state,
)
from .tasks import TaskManager, Task, TaskStatus, get_task_manager
from .bridge import (
    async_start_backend,
    async_stop_backend,
    async_stop_all,
    async_status,
    async_list_models,
    async_get_backend_state,
    async_load_model,
    async_download_model,
    async_unload_model,
    shutdown as bridge_shutdown,
)


__all__ = [
    # Interface
    "Backend",
    "BackendState",
    "Model",
    "Node",
    "ClusterConfig",
    # Registry
    "get_registry",
    "get_backend",
    "get_all_backends",
    "get_active_backend",
    # Control (sync)
    "start_backend",
    "stop_backend",
    "stop_all",
    "status",
    # Models (sync)
    "list_models",
    "download_model",
    "load_model",
    "unload_model",
    "get_loaded_model",
    "get_backend_state",
    # Tasks
    "TaskManager",
    "Task",
    "TaskStatus",
    "get_task_manager",
    # Bridge (async)
    "async_start_backend",
    "async_stop_backend",
    "async_stop_all",
    "async_status",
    "async_list_models",
    "async_get_backend_state",
    "async_load_model",
    "async_download_model",
    "async_unload_model",
    "bridge_shutdown",
]
