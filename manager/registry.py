"""Backend registry - single source of truth for available backends.

Availability is checked once at startup and cached. We don't need to handle
backends being installed while the app is running - just restart to pick them up.
"""

import logging
from typing import Iterator

from .interface import Backend
from .backends import LlamaBackend, FoundryBackend, VllmBackend, SglangBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry of available backends.

    Caches availability at registration time - availability doesn't change
    at runtime (if you install a backend, restart the app).
    """

    def __init__(self):
        self._backends: dict[str, Backend] = {}
        self._availability: dict[str, bool] = {}  # Cached at registration
        self._unavailable_reasons: dict[str, str | None] = {}  # Cached at registration

    def register(self, backend: Backend):
        """Register a backend and cache its availability."""
        self._backends[backend.name] = backend
        # Cache availability check at registration time (only done once)
        available = backend.is_available()
        self._availability[backend.name] = available
        self._unavailable_reasons[backend.name] = backend.get_unavailable_reason()
        logger.debug(f"Registered backend: {backend.name} (available={available})")

    def get(self, name: str) -> Backend | None:
        """Get a backend by name."""
        return self._backends.get(name)

    def get_all(self) -> dict[str, Backend]:
        """Get all registered backends."""
        return self._backends.copy()

    def is_available(self, name: str) -> bool:
        """Check if a backend is available (cached at startup)."""
        return self._availability.get(name, False)

    def get_unavailable_reason(self, name: str) -> str | None:
        """Get reason why backend is unavailable (cached at startup)."""
        return self._unavailable_reasons.get(name)

    def get_available(self) -> dict[str, Backend]:
        """Get only available backends (uses cached availability)."""
        return {
            name: b for name, b in self._backends.items()
            if self._availability.get(name, False)
        }

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
