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

# Backwards compatibility alias
BACKENDS = property(lambda self: get_all_backends())

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
    # Control
    "start_backend",
    "stop_backend",
    "stop_all",
    "status",
    # Models
    "list_models",
    "download_model",
    "load_model",
    "unload_model",
    "get_loaded_model",
    "get_backend_state",
]
