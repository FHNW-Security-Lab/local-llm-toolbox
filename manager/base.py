"""Base backend implementation with sensible defaults."""

from pathlib import Path
import platform
import shutil
import subprocess

import httpx

from .interface import (
    Backend,
    BackendState,
    BackendStatus,
    ClusterConfig,
    Model,
    ModelStatus,
    Node,
    NodeStatus,
)


def _detect_platform() -> str:
    """Detect current platform: 'darwin', 'linux', 'windows'."""
    return platform.system().lower()


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return False


def _has_metal() -> bool:
    """Check if Apple Metal is available (macOS with Apple Silicon)."""
    if _detect_platform() != "darwin":
        return False
    return platform.machine() == "arm64"


# Cache detection results
PLATFORM = _detect_platform()
HAS_CUDA: bool | None = None
HAS_METAL: bool | None = None


def get_platform() -> str:
    return PLATFORM


def has_cuda() -> bool:
    global HAS_CUDA
    if HAS_CUDA is None:
        HAS_CUDA = _has_cuda()
    return HAS_CUDA


def has_metal() -> bool:
    global HAS_METAL
    if HAS_METAL is None:
        HAS_METAL = _has_metal()
    return HAS_METAL


class BaseBackend:
    """
    Base implementation of Backend interface.

    Subclasses should override methods as needed.
    This provides sensible defaults for optional features.
    """

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        api_base: str,
        health_url: str,
        model_format: str = "",
        platforms: list[str] | None = None,
        requires_gpu: str = "",  # "cuda", "metal", or ""
        work_dir: Path | None = None,
    ):
        self._name = name
        self._display_name = display_name
        self._description = description
        self._api_base = api_base
        self._health_url = health_url
        self._model_format = model_format
        self._platforms = platforms or []
        self._requires_gpu = requires_gpu
        self._work_dir = work_dir

    # ─────────────────────────────────────────────────────────────────
    # Metadata
    # ─────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def api_base(self) -> str:
        return self._api_base

    @property
    def model_format(self) -> str:
        return self._model_format

    # ─────────────────────────────────────────────────────────────────
    # Availability checks
    # ─────────────────────────────────────────────────────────────────

    def _is_platform_supported(self) -> bool:
        if not self._platforms:
            return True
        return get_platform() in self._platforms

    def _is_hardware_supported(self) -> bool:
        if not self._requires_gpu:
            return True
        if self._requires_gpu == "cuda":
            return has_cuda()
        if self._requires_gpu == "metal":
            return has_metal()
        return False

    def _is_installed(self) -> bool:
        if self._work_dir is None:
            return True
        return self._work_dir.exists()

    def is_available(self) -> bool:
        return (
            self._is_platform_supported()
            and self._is_hardware_supported()
            and self._is_installed()
        )

    def get_unavailable_reason(self) -> str | None:
        if not self._is_platform_supported():
            supported = ", ".join(self._platforms)
            return f"Requires {supported} (current: {get_platform()})"
        if not self._is_hardware_supported():
            return f"Requires {self._requires_gpu.upper()} GPU"
        if not self._is_installed():
            return "Not installed"
        return None

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle management (subclasses should override)
    # ─────────────────────────────────────────────────────────────────

    def start(self) -> tuple[bool, str]:
        return False, "Not implemented"

    def stop(self) -> tuple[bool, str]:
        return False, "Not implemented"

    def get_state(self) -> BackendState:
        if self.is_healthy():
            return BackendState(
                status=BackendStatus.RUNNING,
                nodes=self.get_cluster_nodes(),
            )
        return BackendState(status=BackendStatus.STOPPED)

    def is_healthy(self, timeout: float = 2.0) -> bool:
        if not self.is_available():
            return False
        try:
            resp = httpx.get(self._health_url, timeout=timeout)
            return resp.status_code < 500
        except httpx.RequestError:
            return False

    # ─────────────────────────────────────────────────────────────────
    # Model management (subclasses should override for full support)
    # ─────────────────────────────────────────────────────────────────

    def list_models(self) -> list[Model]:
        return []

    def list_downloaded_models(self) -> list[Model]:
        return [m for m in self.list_models() if m.downloaded]

    def download_model(self, model_id: str) -> tuple[bool, str]:
        return False, "Model download not supported for this backend"

    def delete_model(self, model_id: str) -> tuple[bool, str]:
        return False, "Model deletion not supported for this backend"

    def load_model(self, model_id: str) -> tuple[bool, str]:
        return False, "Model loading not supported for this backend"

    def unload_model(self) -> tuple[bool, str]:
        return False, "Model unloading not supported for this backend"

    def get_loaded_model(self) -> Model | None:
        return None

    # ─────────────────────────────────────────────────────────────────
    # Cluster management (default: single local node)
    # ─────────────────────────────────────────────────────────────────

    def supports_cluster(self) -> bool:
        return False

    def get_cluster_nodes(self) -> list[Node]:
        """Default: return single local node."""
        return [
            Node(
                id="local",
                hostname="localhost",
                role="local",
                status=NodeStatus.ONLINE if self.is_healthy() else NodeStatus.OFFLINE,
            )
        ]

    def configure_cluster(self, config: ClusterConfig) -> tuple[bool, str]:
        return False, "Clustering not supported for this backend"

    def connect_node(self, node_id: str) -> tuple[bool, str]:
        return False, "Clustering not supported for this backend"

    def disconnect_node(self, node_id: str) -> tuple[bool, str]:
        return False, "Clustering not supported for this backend"
