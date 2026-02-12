"""Base backend implementation with sensible defaults."""

import logging
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

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Platform & GPU detection
# ─────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────
# System stats collection
# ─────────────────────────────────────────────────────────────────


def collect_system_stats() -> dict:
    """Collect memory, GPU, and CPU stats from the local machine.

    Used by both the main node and RPC workers to report system stats.

    Returns:
        Dict with keys: gpu_name, gpu_busy_percent, gpu_memory_used,
                        gpu_memory_total, cpu_percent, memory_used, memory_total
    """
    import os

    stats = {
        "gpu_name": "",
        "gpu_busy_percent": 0,
        "gpu_memory_used": 0,
        "gpu_memory_total": 0,
        "cpu_percent": 0.0,
        "memory_used": 0,
        "memory_total": 0,
    }

    # CPU utilization (cross-platform via os.getloadavg / nproc)
    try:
        load_1min = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        stats["cpu_percent"] = min(round(load_1min / cpu_count * 100, 1), 100.0)
    except (OSError, AttributeError):
        pass

    try:
        if platform.system() == "Darwin":
            stats["gpu_name"] = "Metal" if _has_metal() else "CPU"

            # macOS: total memory via sysctl
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                stats["memory_total"] = int(result.stdout.strip())
                # Don't set gpu_memory on macOS - unified memory makes it meaningless

            # Memory usage via vm_stat
            # Match Activity Monitor: active + inactive + wired + compressor-occupied
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                page_size = 16384
                page_counts = {}
                for line in result.stdout.split("\n"):
                    if "page size of" in line:
                        page_size = int(line.split()[-2])
                    elif ":" in line:
                        val = line.split(":")[1].strip().rstrip(".")
                        if val.isdigit():
                            page_counts[line.split(":")[0].strip()] = int(val)
                used_pages = (
                    page_counts.get("Pages active", 0)
                    + page_counts.get("Pages inactive", 0)
                    + page_counts.get("Pages wired down", 0)
                    + page_counts.get("Pages occupied by compressor", 0)
                )
                stats["memory_used"] = used_pages * page_size

        elif platform.system() == "Linux":
            # Memory from /proc/meminfo
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        stats["memory_total"] = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        stats["memory_used"] = stats["memory_total"] - int(line.split()[1]) * 1024
                        break

            # GPU: try NVIDIA first, then sysfs (AMD/Intel integrated)
            if shutil.which("nvidia-smi"):
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 3:
                        stats["gpu_name"] = parts[0].strip()
                        stats["gpu_memory_used"] = int(parts[1].strip()) * 1024 * 1024
                        stats["gpu_memory_total"] = int(parts[2].strip()) * 1024 * 1024
                    if len(parts) >= 4:
                        stats["gpu_busy_percent"] = int(parts[3].strip())
            else:
                # Try sysfs for integrated GPUs (AMD amdgpu, Intel i915/xe)
                gpu_info = _detect_gpu_sysfs()
                if gpu_info:
                    stats["gpu_name"] = gpu_info["name"]
                    stats["gpu_busy_percent"] = gpu_info["busy_percent"]
                    stats["gpu_memory_used"] = gpu_info.get("memory_used", 0)
                    stats["gpu_memory_total"] = gpu_info.get("memory_total", 0)

    except Exception as e:
        logger.debug(f"Failed to collect system stats: {e}")

    # Fall back to "CPU" if no GPU was detected
    if not stats["gpu_name"]:
        stats["gpu_name"] = "CPU"

    return stats


def _detect_gpu_sysfs() -> dict | None:
    """Detect GPU info via Linux sysfs (DRM subsystem).

    Works for AMD (amdgpu), Intel (i915/xe), and other DRM-based drivers.

    Returns:
        Dict with name, busy_percent, memory_used, memory_total, or None if no GPU found.
    """
    drm_path = Path("/sys/class/drm")
    if not drm_path.exists():
        return None

    for card_dir in sorted(drm_path.glob("card[0-9]*")):
        device_dir = card_dir / "device"
        busy_file = device_dir / "gpu_busy_percent"

        if not busy_file.exists():
            continue

        try:
            busy_percent = int(busy_file.read_text().strip())

            # Get driver name and PCI ID from uevent
            driver = ""
            pci_id = ""
            uevent_file = device_dir / "uevent"
            if uevent_file.exists():
                for line in uevent_file.read_text().splitlines():
                    if line.startswith("DRIVER="):
                        driver = line.split("=", 1)[1]
                    elif line.startswith("PCI_ID="):
                        pci_id = line.split("=", 1)[1]

            gpu_name = _get_gpu_name_lspci(driver, pci_id)

            result = {
                "name": gpu_name,
                "busy_percent": busy_percent,
            }

            # Check for VRAM info (available on some AMD GPUs)
            vram_used_file = device_dir / "mem_info_vram_used"
            vram_total_file = device_dir / "mem_info_vram_total"
            if vram_used_file.exists() and vram_total_file.exists():
                result["memory_used"] = int(vram_used_file.read_text().strip())
                result["memory_total"] = int(vram_total_file.read_text().strip())

            return result

        except (ValueError, OSError) as e:
            logger.debug(f"Failed to read GPU info from {card_dir}: {e}")
            continue

    return None


def _get_gpu_name_lspci(driver: str, pci_id: str) -> str:
    """Get a human-readable GPU name via lspci, falling back to driver info."""
    if shutil.which("lspci"):
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "VGA" in line or "3D" in line or "Display" in line:
                        # Format: "00:02.0 VGA compatible controller: AMD ..."
                        parts = line.split(": ", 1)
                        if len(parts) >= 2:
                            return parts[1].strip()
        except Exception:
            pass

    if driver:
        return f"{driver} ({pci_id})" if pci_id else driver
    return "GPU"


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
        healthy = self.is_healthy()
        return BackendState(
            status=BackendStatus.RUNNING if healthy else BackendStatus.STOPPED,
            nodes=self.get_cluster_nodes(healthy),
        )

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

    def get_cluster_nodes(self, is_running: bool | None = None) -> list[Node]:
        """Default: return single local node with system stats.

        Args:
            is_running: If known, whether backend is running (avoids duplicate health check).
        """
        status = NodeStatus.ONLINE if is_running else NodeStatus.OFFLINE
        return [
            Node(
                id="local",
                hostname="localhost",
                role="local",
                status=status,
                **collect_system_stats(),
            )
        ]

    def configure_cluster(self, config: ClusterConfig) -> tuple[bool, str]:
        return False, "Clustering not supported for this backend"

    def connect_node(self, node_id: str) -> tuple[bool, str]:
        return False, "Clustering not supported for this backend"

    def disconnect_node(self, node_id: str) -> tuple[bool, str]:
        return False, "Clustering not supported for this backend"
