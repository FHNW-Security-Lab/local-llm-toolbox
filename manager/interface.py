"""Abstract interface that all backends must implement."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class BackendStatus(str, Enum):
    """Backend lifecycle status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class ModelStatus(str, Enum):
    """Model loading status."""
    IDLE = "idle"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class NodeStatus(str, Enum):
    """Cluster node status."""
    ONLINE = "online"
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ERROR = "error"


@dataclass
class Model:
    """Represents a model that can be loaded."""
    id: str
    name: str
    size_bytes: int = 0
    format: str = ""  # "gguf", "onnx", "safetensors"
    quantization: str = ""  # "Q4_K_M", "Q8_0", etc.
    downloaded: bool = False
    source: str = ""  # URL or identifier for downloading
    task: str = ""  # "chat-completions", "automatic-speech-recognition", "embeddings"


@dataclass
class Node:
    """Represents a node in a cluster."""
    id: str
    hostname: str
    role: str = "local"  # "local", "remote"
    status: NodeStatus = NodeStatus.OFFLINE
    gpu_name: str = ""
    gpu_memory_used: int = 0  # bytes
    gpu_memory_total: int = 0  # bytes
    cpu_percent: float = 0.0
    memory_used: int = 0  # bytes
    memory_total: int = 0  # bytes


@dataclass
class ClusterConfig:
    """Configuration for setting up a cluster."""
    nodes: list[dict] = field(default_factory=list)
    # Each node dict: {"hostname": str, "user": str, "gpu_layers": int, ...}


@dataclass
class BackendState:
    """Complete state of a backend."""
    status: BackendStatus = BackendStatus.STOPPED
    error: str | None = None
    # Model state
    model_status: ModelStatus = ModelStatus.IDLE
    loaded_model: Model | None = None
    model_error: str | None = None
    # Cluster state
    nodes: list[Node] = field(default_factory=list)


@runtime_checkable
class Backend(Protocol):
    """Protocol that all backends must implement."""

    # ─────────────────────────────────────────────────────────────────
    # Metadata (properties, no side effects)
    # ─────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Unique identifier for this backend."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        ...

    @property
    def description(self) -> str:
        """Description of the backend."""
        ...

    @property
    def api_base(self) -> str:
        """Base URL for the inference API when running."""
        ...

    @property
    def model_format(self) -> str:
        """Model format this backend uses (gguf, onnx, safetensors)."""
        ...

    # ─────────────────────────────────────────────────────────────────
    # Availability checks
    # ─────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if backend can be used on this system."""
        ...

    def get_unavailable_reason(self) -> str | None:
        """Return reason why unavailable, or None if available."""
        ...

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle management
    # ─────────────────────────────────────────────────────────────────

    def start(self) -> tuple[bool, str]:
        """Start the backend. Returns (success, message)."""
        ...

    def stop(self) -> tuple[bool, str]:
        """Stop the backend. Returns (success, message)."""
        ...

    def get_state(self) -> BackendState:
        """Get current state of the backend."""
        ...

    def is_healthy(self) -> bool:
        """Quick health check - is the backend responding?"""
        ...

    # ─────────────────────────────────────────────────────────────────
    # Model management
    # ─────────────────────────────────────────────────────────────────

    def list_models(self) -> list[Model]:
        """List all available models (downloaded + downloadable)."""
        ...

    def list_downloaded_models(self) -> list[Model]:
        """List only downloaded/cached models."""
        ...

    def download_model(self, model_id: str) -> tuple[bool, str]:
        """Download a model. Returns (success, message)."""
        ...

    def delete_model(self, model_id: str) -> tuple[bool, str]:
        """Delete a downloaded model. Returns (success, message)."""
        ...

    def load_model(self, model_id: str) -> tuple[bool, str]:
        """Load a model for inference. Returns (success, message)."""
        ...

    def unload_model(self) -> tuple[bool, str]:
        """Unload the current model. Returns (success, message)."""
        ...

    def get_loaded_model(self) -> Model | None:
        """Get the currently loaded model, or None."""
        ...

    # ─────────────────────────────────────────────────────────────────
    # Cluster management (optional - return sensible defaults if not supported)
    # ─────────────────────────────────────────────────────────────────

    def supports_cluster(self) -> bool:
        """Whether this backend supports multi-node clustering."""
        ...

    def get_cluster_nodes(self) -> list[Node]:
        """Get all nodes in the cluster (including local)."""
        ...

    def configure_cluster(self, config: ClusterConfig) -> tuple[bool, str]:
        """Configure cluster nodes. Returns (success, message)."""
        ...

    def connect_node(self, node_id: str) -> tuple[bool, str]:
        """Connect to a specific node. Returns (success, message)."""
        ...

    def disconnect_node(self, node_id: str) -> tuple[bool, str]:
        """Disconnect a specific node. Returns (success, message)."""
        ...
