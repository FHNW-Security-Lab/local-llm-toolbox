"""In-memory runtime state for the manager."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RuntimeState:
    """Runtime state for a single backend (in-memory only).

    Note: This is different from interface.BackendState which is the full
    state returned by backends. This tracks minimal runtime info.
    """
    loaded_model: str | None = None
    cluster_config: dict = field(default_factory=dict)


@dataclass
class ManagerState:
    """Global manager state (in-memory only)."""
    active_backend: str | None = None
    backends: dict[str, RuntimeState] = field(default_factory=dict)

    def get_backend_state(self, name: str) -> RuntimeState:
        """Get or create state for a backend."""
        if name not in self.backends:
            self.backends[name] = RuntimeState()
        return self.backends[name]


class StateStore:
    """In-memory state storage."""

    def __init__(self):
        self._state = ManagerState()

    @property
    def state(self) -> ManagerState:
        return self._state

    def set_active_backend(self, name: str | None):
        """Set the active backend."""
        logger.debug(f"Setting active backend: {name}")
        self._state.active_backend = name

    def set_loaded_model(self, backend: str, model: str | None):
        """Set the loaded model for a backend."""
        logger.debug(f"Setting loaded model for {backend}: {model}")
        self._state.get_backend_state(backend).loaded_model = model

    def get_loaded_model(self, backend: str) -> str | None:
        """Get the loaded model for a backend."""
        return self._state.get_backend_state(backend).loaded_model

    def set_cluster_config(self, backend: str, config: dict):
        """Store cluster configuration for a backend."""
        logger.debug(f"Setting cluster config for {backend}: {config}")
        self._state.get_backend_state(backend).cluster_config = config

    def get_cluster_config(self, backend: str) -> dict:
        """Get cluster configuration for a backend."""
        return self._state.get_backend_state(backend).cluster_config

    def clear_backend_state(self, backend: str):
        """Clear state for a backend."""
        logger.debug(f"Clearing state for {backend}")
        if backend in self._state.backends:
            self._state.backends[backend] = RuntimeState()


# Global state store instance
_store: StateStore | None = None


def get_store() -> StateStore:
    """Get the global state store."""
    global _store
    if _store is None:
        _store = StateStore()
    return _store
