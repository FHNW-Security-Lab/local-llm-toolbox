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
    logger.debug("Getting status of all backends")
    registry = get_registry()
    store = get_store()
    active_name = store.state.active_backend

    # Get the active backend from state store - trust state, don't require health check
    # This is important for backends like Foundry that use dynamic ports
    active = registry.get(active_name) if active_name else None

    logger.debug(f"Status: active_backend={active_name}")
    return Status(
        active=active,
        conflict=False,
        backends={name: name == active_name for name, b in registry.get_all().items()},
    )


def stop_backend(backend: Backend, timeout: float = 15.0) -> bool:
    """Stop a backend. Returns True if stopped successfully.

    This function trusts the backend's stop() implementation. Most backends
    (like Foundry's CLI or llama's process manager) block until stopped,
    so we don't need expensive health check polling.
    """
    logger.info(f"Stopping backend: {backend.name}")
    store = get_store()

    # Clear active backend state first (before any expensive checks)
    was_active = store.state.active_backend == backend.name
    if was_active:
        logger.debug(f"Clearing active backend state for {backend.name}")
        store.set_active_backend(None)

    # If this backend wasn't marked as active in our state, it's likely not running.
    # Skip expensive operations. Just call stop() directly (best effort).
    if not was_active:
        logger.debug(f"Backend {backend.name} was not active, calling stop() directly")
        backend.stop()  # Best effort, ignore result
        return True

    # Stop the backend - trust its stop() implementation
    success, msg = backend.stop()
    if success:
        logger.info(f"Backend {backend.name} stopped successfully")
    else:
        logger.warning(f"Backend {backend.name} stop() returned failure: {msg}")
    return success


def stop_all() -> bool:
    """Stop all running backends. Returns True if all stopped.

    Only stops the backend marked as active in our state store.
    We don't scan all backends for "orphans" - that triggers expensive
    SDK initialization (e.g., Foundry). If you have orphan processes,
    stop them manually.
    """
    store = get_store()
    registry = get_registry()

    active_name = store.state.active_backend
    if not active_name:
        logger.info("No active backend to stop")
        return True

    backend = registry.get(active_name)
    if not backend:
        logger.warning(f"Active backend {active_name} not found in registry")
        store.set_active_backend(None)
        return True

    logger.info(f"Stopping active backend: {active_name}")
    return stop_backend(backend)


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
    logger.info(f"Starting backend: {name} (force={force})")
    registry = get_registry()

    if name not in registry:
        available = ", ".join(registry.get_all().keys())
        logger.warning(f"Unknown backend: {name}")
        return False, f"Unknown backend: {name}. Available: {available}"

    backend = registry.get(name)
    if not backend:
        logger.error(f"Backend not found in registry: {name}")
        return False, f"Backend not found: {name}"

    # Check availability
    reason = backend.get_unavailable_reason()
    if reason:
        logger.warning(f"Backend {name} unavailable: {reason}")
        return False, f"{backend.display_name}: {reason}"

    store = get_store()

    # Check for already active backend
    current_active = store.state.active_backend
    if current_active == name:
        logger.debug(f"Backend {name} is already active")
        return True, f"{backend.display_name} is already active"

    # Check if we need to stop the currently active backend
    if current_active:
        if not force:
            current = registry.get(current_active)
            logger.debug(f"Cannot start {name}: {current_active} is active")
            return False, f"{current.display_name if current else current_active} is active. Use --force to switch."

        # Stop the active backend
        current = registry.get(current_active)
        if current:
            logger.info(f"Force mode: stopping active backend {current_active}")
            stop_backend(current)

    # Start the backend
    logger.debug(f"Calling backend.start() for {name}")
    success, msg = backend.start()
    if not success:
        logger.warning(f"Backend {name} failed to start: {msg}")
        return False, msg

    # Mark as active
    store.set_active_backend(name)
    logger.info(f"Backend {name} started successfully: {msg}")

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
    logger.debug(f"Listing models for backend: {name or '(none)'}")
    if not name:
        return {}

    backend = registry.get(name)
    if not backend or not registry.is_available(name):
        logger.debug(f"Backend {name} not available for listing models")
        return {}

    try:
        models = backend.list_models()
        logger.debug(f"Found {len(models)} models for {name}")
        return {name: models}
    except Exception as e:
        logger.warning(f"Failed to list models for {name}: {e}")
        return {name: []}


def download_model(backend_name: str, model_id: str) -> tuple[bool, str]:
    """Download a model for a backend."""
    logger.info(f"Downloading model {model_id} for {backend_name}")
    registry = get_registry()
    backend = registry.get(backend_name)
    if not backend:
        logger.warning(f"Unknown backend: {backend_name}")
        return False, f"Unknown backend: {backend_name}"
    if not registry.is_available(backend_name):
        reason = registry.get_unavailable_reason(backend_name) or "Not available"
        logger.warning(f"Backend {backend_name} unavailable: {reason}")
        return False, reason
    result = backend.download_model(model_id)
    logger.info(f"Download result for {model_id}: {result}")
    return result


def load_model(backend_name: str, model_id: str) -> tuple[bool, str]:
    """Load a model on a backend."""
    logger.info(f"Loading model {model_id} on {backend_name}")
    registry = get_registry()
    backend = registry.get(backend_name)
    if not backend:
        logger.warning(f"Unknown backend: {backend_name}")
        return False, f"Unknown backend: {backend_name}"
    if not registry.is_available(backend_name):
        reason = registry.get_unavailable_reason(backend_name) or "Not available"
        logger.warning(f"Backend {backend_name} unavailable: {reason}")
        return False, reason
    result = backend.load_model(model_id)
    if result[0]:
        logger.info(f"Model {model_id} loaded successfully on {backend_name}")
    else:
        logger.warning(f"Failed to load model {model_id} on {backend_name}: {result[1]}")
    return result


def unload_model(backend_name: str) -> tuple[bool, str]:
    """Unload the current model from a backend."""
    logger.info(f"Unloading model from {backend_name}")
    backend = get_registry().get(backend_name)
    if not backend:
        logger.warning(f"Unknown backend: {backend_name}")
        return False, f"Unknown backend: {backend_name}"
    result = backend.unload_model()
    if result[0]:
        logger.info(f"Model unloaded from {backend_name}")
    else:
        logger.warning(f"Failed to unload from {backend_name}: {result[1]}")
    return result


def get_loaded_model(backend_name: str) -> Model | None:
    """Get the currently loaded model for a backend."""
    logger.debug(f"Getting loaded model for {backend_name}")
    backend = get_registry().get(backend_name)
    if not backend:
        return None
    return backend.get_loaded_model()


def get_backend_state(backend_name: str) -> BackendState | None:
    """Get the full state of a backend."""
    logger.debug(f"Getting backend state for {backend_name}")
    backend = get_registry().get(backend_name)
    if not backend:
        return None
    return backend.get_state()
