"""Backend registry - single source of truth for available backends."""

import logging
from typing import Iterator

from .interface import Backend
from .backends import LlamaBackend, FoundryBackend, VllmBackend, SglangBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry of available backends."""

    def __init__(self):
        self._backends: dict[str, Backend] = {}

    def register(self, backend: Backend):
        """Register a backend."""
        self._backends[backend.name] = backend
        logger.debug(f"Registered backend: {backend.name}")

    def get(self, name: str) -> Backend | None:
        """Get a backend by name."""
        return self._backends.get(name)

    def get_all(self) -> dict[str, Backend]:
        """Get all registered backends."""
        return self._backends.copy()

    def get_available(self) -> dict[str, Backend]:
        """Get only available backends."""
        return {
            name: b for name, b in self._backends.items()
            if b.is_available()
        }

    def get_active(self) -> Backend | None:
        """Get the currently active (healthy) backend."""
        active = [b for b in self._backends.values() if b.is_healthy()]
        if len(active) == 1:
            return active[0]
        return None

    def get_all_active(self) -> list[Backend]:
        """Get all active backends (should be 0 or 1)."""
        return [b for b in self._backends.values() if b.is_healthy()]

    def __iter__(self) -> Iterator[Backend]:
        return iter(self._backends.values())

    def __len__(self) -> int:
        return len(self._backends)

    def __contains__(self, name: str) -> bool:
        return name in self._backends


def create_default_registry() -> BackendRegistry:
    """Create registry with default backends."""
    registry = BackendRegistry()

    # Register all backends
    registry.register(LlamaBackend())
    registry.register(FoundryBackend())
    registry.register(VllmBackend())
    registry.register(SglangBackend())

    return registry


# Global registry instance
_registry: BackendRegistry | None = None


def get_registry() -> BackendRegistry:
    """Get the global backend registry."""
    global _registry
    if _registry is None:
        _registry = create_default_registry()
    return _registry


# Convenience accessors (for backwards compatibility)
def get_backend(name: str) -> Backend | None:
    """Get a backend by name."""
    return get_registry().get(name)


def get_all_backends() -> dict[str, Backend]:
    """Get all backends."""
    return get_registry().get_all()


def get_active_backend() -> Backend | None:
    """Get the currently active backend from state store.

    Returns the backend if it's marked as active, without requiring
    a health check. This is important for backends like Foundry that
    use dynamic ports - the SDK knows the correct endpoint even if
    is_healthy() might fail due to port changes.
    """
    from .state import get_store
    store = get_store()
    active_name = store.state.active_backend
    if not active_name:
        return None
    return get_registry().get(active_name)
