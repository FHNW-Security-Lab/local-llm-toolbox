"""Start/stop/status control for backends."""

import logging
import time
from dataclasses import dataclass

from .interface import Backend, BackendState, Model
from .registry import get_registry, get_active_backend, get_all_backends
from .state import get_store

logger = logging.getLogger(__name__)


@dataclass
class Status:
    """Current status of the toolbox."""
    active: Backend | None
    conflict: bool  # True if multiple backends running
    backends: dict[str, bool]  # name -> is_active (user selected this backend)


def status() -> Status:
    """Get current status of all backends."""
    registry = get_registry()
    store = get_store()
    active_name = store.state.active_backend

    # Get the active backend from state store - trust state, don't require health check
    # This is important for backends like Foundry that use dynamic ports
    active = registry.get(active_name) if active_name else None

    return Status(
        active=active,
        conflict=False,
        backends={name: name == active_name for name, b in registry.get_all().items()},
    )


def stop_backend(backend: Backend, timeout: float = 15.0) -> bool:
    """Stop a backend. Returns True if stopped successfully."""
    store = get_store()

    # Clear active backend state
    if store.state.active_backend == backend.name:
        store.set_active_backend(None)

    if not backend.is_healthy():
        return True  # Already stopped

    success, msg = backend.stop()
    if not success:
        return False

    # Wait for it to actually stop
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not backend.is_healthy():
            return True
        time.sleep(0.5)

    return not backend.is_healthy()


def stop_all() -> bool:
    """Stop all running backends. Returns True if all stopped."""
    store = get_store()
    registry = get_registry()
    success = True

    # First, stop the backend marked as active in our state
    active_name = store.state.active_backend
    if active_name:
        backend = registry.get(active_name)
        if backend:
            logger.info(f"Stopping active backend: {active_name}")
            if not stop_backend(backend):
                success = False

    # Also check for any backends that are healthy but not in our state
    # (e.g., orphaned processes from previous runs)
    for backend in registry.get_all().values():
        if backend.is_healthy():
            logger.info(f"Stopping orphaned backend: {backend.name}")
            if not stop_backend(backend):
                success = False

    return success


def start_backend(
    name: str,
    *,
    force: bool = False,
    wait_timeout: float = 30.0,
) -> tuple[bool, str]:
    """
    Start a backend.

    Args:
        name: Backend name (e.g., "llama")
        force: If True, stop any running backend first
        wait_timeout: How long to wait for backend to become healthy

    Returns:
        (success, message) tuple
    """
    registry = get_registry()

    if name not in registry:
        available = ", ".join(registry.get_all().keys())
        return False, f"Unknown backend: {name}. Available: {available}"

    backend = registry.get(name)
    if not backend:
        return False, f"Backend not found: {name}"

    # Check availability
    reason = backend.get_unavailable_reason()
    if reason:
        return False, f"{backend.display_name}: {reason}"

    store = get_store()

    # Check for already active backend
    current_active = store.state.active_backend
    if current_active == name:
        return True, f"{backend.display_name} is already active"

    # Stop any running backends
    running = [b for b in registry.get_all().values() if b.is_healthy()]
    if running or current_active:
        if not force:
            if running:
                names = ", ".join(b.display_name for b in running)
                return False, f"Backend running: {names}. Use --force to stop."
            if current_active:
                current = registry.get(current_active)
                return False, f"{current.display_name if current else current_active} is active. Use --force to switch."

        # Stop running backends
        for b in running:
            stop_backend(b)

    # Start the backend
    success, msg = backend.start()
    if not success:
        return False, msg

    # Mark as active
    store.set_active_backend(name)

    return True, msg


# ─────────────────────────────────────────────────────────────────
# Model Management
# ─────────────────────────────────────────────────────────────────

def list_models(backend_name: str | None = None) -> dict[str, list[Model]]:
    """
    List models for the active backend only.

    Args:
        backend_name: Specific backend, or None for active backend

    Returns:
        Dict mapping backend name to list of models
    """
    store = get_store()
    registry = get_registry()

    # Determine which backend to query
    name = backend_name or store.state.active_backend
    if not name:
        return {}

    backend = registry.get(name)
    if not backend or not backend.is_available():
        return {}

    try:
        models = backend.list_models()
        return {name: models}
    except Exception as e:
        logger.warning(f"Failed to list models for {name}: {e}")
        return {name: []}


def download_model(backend_name: str, model_id: str) -> tuple[bool, str]:
    """Download a model for a backend."""
    backend = get_registry().get(backend_name)
    if not backend:
        return False, f"Unknown backend: {backend_name}"
    if not backend.is_available():
        return False, backend.get_unavailable_reason() or "Not available"
    return backend.download_model(model_id)


def load_model(backend_name: str, model_id: str) -> tuple[bool, str]:
    """Load a model on a backend."""
    backend = get_registry().get(backend_name)
    if not backend:
        return False, f"Unknown backend: {backend_name}"
    if not backend.is_available():
        return False, backend.get_unavailable_reason() or "Not available"
    return backend.load_model(model_id)


def unload_model(backend_name: str) -> tuple[bool, str]:
    """Unload the current model from a backend."""
    backend = get_registry().get(backend_name)
    if not backend:
        return False, f"Unknown backend: {backend_name}"
    return backend.unload_model()


def get_loaded_model(backend_name: str) -> Model | None:
    """Get the currently loaded model for a backend."""
    backend = get_registry().get(backend_name)
    if not backend:
        return None
    return backend.get_loaded_model()


def get_backend_state(backend_name: str) -> BackendState | None:
    """Get the full state of a backend."""
    backend = get_registry().get(backend_name)
    if not backend:
        return None
    return backend.get_state()
