"""In-memory runtime state for the manager."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BackendState:
    """Runtime state for a single backend."""
    loaded_model: str | None = None


@dataclass
class ManagerState:
    """Global manager state (in-memory only)."""
    active_backend: str | None = None
    backends: dict[str, BackendState] = field(default_factory=dict)

    def get_backend_state(self, name: str) -> BackendState:
        """Get or create state for a backend."""
        if name not in self.backends:
            self.backends[name] = BackendState()
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
        self._state.active_backend = name

    def set_loaded_model(self, backend: str, model: str | None):
        """Set the loaded model for a backend."""
        self._state.get_backend_state(backend).loaded_model = model

    def get_loaded_model(self, backend: str) -> str | None:
        """Get the loaded model for a backend."""
        return self._state.get_backend_state(backend).loaded_model

    def clear_backend_state(self, backend: str):
        """Clear state for a backend."""
        if backend in self._state.backends:
            self._state.backends[backend] = BackendState()


# Global state store instance
_store: StateStore | None = None


def get_store() -> StateStore:
    """Get the global state store."""
    global _store
    if _store is None:
        _store = StateStore()
    return _store
