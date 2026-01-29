"""Microsoft Foundry Local backend implementation."""

import logging
import shutil

from ..base import BaseBackend
from ..config import FoundrySettings, get_foundry_settings
from ..interface import (
    BackendState,
    BackendStatus,
    Model,
    ModelStatus,
    Node,
    NodeStatus,
)
from ..process import check_health_sync

logger = logging.getLogger(__name__)

# Lazy import SDK to avoid ImportError if not installed
_foundry_sdk = None


def _get_sdk():
    """Lazy load the Foundry SDK."""
    global _foundry_sdk
    if _foundry_sdk is None:
        try:
            from foundry_local import FoundryLocalManager
            _foundry_sdk = FoundryLocalManager
        except ImportError:
            _foundry_sdk = False
    return _foundry_sdk if _foundry_sdk else None


class FoundryBackend(BaseBackend):
    """
    Microsoft Foundry Local backend.

    Uses ONNX models with ONNX Runtime optimization.
    Available on macOS and Windows.

    Install:
    - macOS: brew install microsoft/foundrylocal/foundrylocal
    - Windows: winget install Microsoft.FoundryLocal
    - SDK: pip install foundry-local-sdk

    Configuration via environment variables (prefix FOUNDRY_):
    - FOUNDRY_PORT: Server port (default: 5273)
    - FOUNDRY_DOWNLOAD_TIMEOUT: Download timeout in seconds (default: 14400)
    - FOUNDRY_LOAD_TIMEOUT: Model load timeout in seconds (default: 1800)
    """

    def __init__(self, settings: FoundrySettings | None = None):
        self.settings = settings or get_foundry_settings()
        self._manager = None  # Lazy SDK manager
        super().__init__(
            name="foundry",
            display_name="Foundry Local",
            description=(
                "Microsoft AI Foundry Local. ONNX Runtime optimized inference. "
                "Install: brew install microsoft/foundrylocal/foundrylocal (macOS) "
                "or winget install Microsoft.FoundryLocal (Windows)."
            ),
            api_base="",  # Dynamic - see api_base property
            health_url="",  # Not used - SDK handles health
            model_format="onnx",
            platforms=["darwin", "windows"],
        )

    @property
    def api_base(self) -> str:
        """Get API base URL from SDK (Foundry uses dynamic ports).

        Returns the base URL without /v1 suffix for consistency with other backends.
        The SDK's manager.endpoint includes /v1, so we strip it.
        """
        manager = self._get_manager()
        if manager:
            try:
                endpoint = manager.endpoint
                # SDK returns endpoint with /v1 suffix (e.g., http://127.0.0.1:56072/v1)
                # Strip it for consistency with other backends
                if endpoint.endswith("/v1"):
                    return endpoint[:-3]
                return endpoint
            except Exception as e:
                logger.debug(f"Failed to get endpoint from SDK: {e}")
        return f"http://localhost:{self.settings.port}"

    def _get_manager(self, ensure_service: bool = False):
        """Get or create the SDK manager (lazy init).

        Args:
            ensure_service: If True, start the service if not running (needed for catalog ops)
        """
        if self._manager is None:
            sdk_class = _get_sdk()
            if sdk_class:
                # Don't auto-bootstrap - we control when to start
                self._manager = sdk_class(bootstrap=False)

        # Some SDK operations need the service running
        if ensure_service and self._manager:
            try:
                if not self._manager.is_service_running():
                    self._manager.start_service()
            except Exception as e:
                logger.debug(f"Failed to ensure service: {e}")

        return self._manager

    # ─────────────────────────────────────────────────────────────────
    # Availability
    # ─────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if not super().is_available():
            return False
        # Need both CLI and SDK
        if shutil.which("foundry") is None:
            return False
        if _get_sdk() is None:
            return False
        return True

    def get_unavailable_reason(self) -> str | None:
        reason = super().get_unavailable_reason()
        if reason:
            return reason
        if shutil.which("foundry") is None:
            return "foundry CLI not installed"
        if _get_sdk() is None:
            return "foundry-local-sdk not installed (pip install foundry-local-sdk)"
        return None

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def start(self) -> tuple[bool, str]:
        """Start the Foundry service."""
        if not self.is_available():
            return False, self.get_unavailable_reason() or "Not available"

        try:
            manager = self._get_manager()
            if manager:
                manager.start_service()
                return True, "Foundry service started"
            return False, "SDK not available"
        except Exception as e:
            return False, f"Failed to start: {e}"

    def stop(self) -> tuple[bool, str]:
        """Stop the Foundry service."""
        import subprocess
        try:
            # SDK doesn't have stop_service, use CLI
            result = subprocess.run(
                ["foundry", "service", "stop"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True, "Foundry service stopped"
            return False, f"Failed to stop: {result.stderr}"
        except Exception as e:
            return False, f"Failed to stop: {e}"

    def is_healthy(self, timeout: float = 2.0) -> bool:
        """Check if Foundry service is responding."""
        manager = self._get_manager()
        if manager:
            try:
                return manager.is_service_running()
            except Exception:
                pass
        return check_health_sync(self._health_url, timeout)

    def get_state(self) -> BackendState:
        """Get current backend state."""
        is_running = self.is_healthy()

        # Get loaded model from SDK
        loaded_model = None
        if is_running:
            manager = self._get_manager()
            if manager:
                try:
                    loaded = manager.list_loaded_models()
                    if loaded:
                        # Convert first loaded model to our Model type
                        info = loaded[0]
                        loaded_model = Model(
                            id=info.alias,
                            name=info.alias,
                            size_bytes=info.file_size_mb * 1024 * 1024 if info.file_size_mb else None,
                            format="onnx",
                            downloaded=True,
                        )
                except Exception as e:
                    logger.debug(f"Failed to get loaded models: {e}")

        return BackendState(
            status=BackendStatus.RUNNING if is_running else BackendStatus.STOPPED,
            model_status=ModelStatus.READY if loaded_model else ModelStatus.IDLE,
            loaded_model=loaded_model,
            nodes=self.get_cluster_nodes(),
        )

    # ─────────────────────────────────────────────────────────────────
    # Model Management
    # ─────────────────────────────────────────────────────────────────

    def list_models(self) -> list[Model]:
        """List available models from Foundry catalog."""
        if not self.is_available():
            return []

        # This starts the service if needed
        manager = self._get_manager(ensure_service=True)
        if not manager:
            return []

        # Get catalog and cached models
        catalog = manager.list_catalog_models()
        cached_aliases = set()
        try:
            cached = manager.list_cached_models()
            cached_aliases = {m.alias for m in cached}
        except Exception:
            pass

        # Deduplicate by alias (catalog has GPU/CPU variants)
        seen_aliases = set()
        models = []
        for info in catalog:
            if info.alias in seen_aliases:
                continue
            seen_aliases.add(info.alias)

            models.append(Model(
                id=info.alias,
                name=info.alias,
                size_bytes=info.file_size_mb * 1024 * 1024 if info.file_size_mb else None,
                format="onnx",
                downloaded=info.alias in cached_aliases,
            ))

        return models

    def download_model(self, model_id: str) -> tuple[bool, str]:
        """Download a model from Foundry catalog."""
        if not self.is_available():
            return False, self.get_unavailable_reason() or "Not available"

        manager = self._get_manager()
        if not manager:
            return False, "SDK not available"

        try:
            info = manager.download_model(model_id)
            return True, f"Downloaded {info.alias}"
        except Exception as e:
            return False, f"Download failed: {e}"

    def load_model(self, model_id: str) -> tuple[bool, str]:
        """Load a model for inference."""
        if not self.is_available():
            return False, self.get_unavailable_reason() or "Not available"

        manager = self._get_manager()
        if not manager:
            return False, "SDK not available"

        try:
            info = manager.load_model(model_id)
            return True, f"Loaded {info.alias}"
        except Exception as e:
            return False, f"Failed to load: {e}"

    def unload_model(self) -> tuple[bool, str]:
        """Unload the current model."""
        manager = self._get_manager()
        if not manager:
            return False, "SDK not available"

        try:
            loaded = manager.list_loaded_models()
            for model in loaded:
                manager.unload_model(model.alias)
            return True, "Model unloaded"
        except Exception as e:
            return False, f"Failed to unload: {e}"

    def get_loaded_model(self) -> Model | None:
        """Get the currently loaded model.

        Returns a Model with:
        - id: The full model ID required by the REST API (e.g., "qwen2.5-0.5b-instruct-generic-gpu:4")
        - name: The alias for display (e.g., "qwen2.5-0.5b")
        """
        manager = self._get_manager()
        if not manager:
            return None

        try:
            loaded = manager.list_loaded_models()
            if loaded:
                info = loaded[0]
                # The info object from list_loaded_models() has .id with the full model ID
                # This is required by the REST API (alias doesn't work for /v1/chat/completions)
                model_id = info.id
                logger.debug(f"Loaded model: alias={info.alias}, id={model_id}")
                return Model(
                    id=model_id,
                    name=info.alias,  # Keep alias as display name
                    size_bytes=info.file_size_mb * 1024 * 1024 if info.file_size_mb else None,
                    format="onnx",
                    downloaded=True,
                )
        except Exception as e:
            logger.debug(f"Failed to get loaded model: {e}")
        return None

    # ─────────────────────────────────────────────────────────────────
    # Cluster (not supported)
    # ─────────────────────────────────────────────────────────────────

    def supports_cluster(self) -> bool:
        return False

    def get_cluster_nodes(self) -> list[Node]:
        return [Node(
            id="local",
            hostname="localhost",
            role="local",
            status=NodeStatus.ONLINE if self.is_healthy() else NodeStatus.OFFLINE,
        )]
