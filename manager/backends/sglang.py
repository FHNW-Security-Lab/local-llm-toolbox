"""SGLang backend implementation (stub)."""

import logging

from ..base import BaseBackend
from ..interface import (
    BackendState,
    BackendStatus,
    Model,
    ModelStatus,
    Node,
    NodeStatus,
)

logger = logging.getLogger(__name__)


class SglangBackend(BaseBackend):
    """
    SGLang backend for fast inference with RadixAttention.

    Requires Linux with CUDA GPU.
    Uses HuggingFace model format (safetensors).

    Not yet implemented - placeholder for future development.
    """

    def __init__(self):
        super().__init__(
            name="sglang",
            display_name="SGLang",
            description=(
                "Fast inference with RadixAttention for efficient KV cache reuse. "
                "Requires Linux with CUDA GPU. Uses HuggingFace models."
            ),
            api_base="http://localhost:30000",
            health_url="http://localhost:30000/health",
            model_format="safetensors",
            platforms=["linux"],
            requires_gpu="cuda",
        )

    def is_available(self) -> bool:
        # Not implemented yet
        return False

    def get_unavailable_reason(self) -> str | None:
        base_reason = super().get_unavailable_reason()
        if base_reason:
            return base_reason
        return "Not implemented"

    def start(self) -> tuple[bool, str]:
        return False, "SGLang backend not yet implemented"

    def stop(self) -> tuple[bool, str]:
        return False, "SGLang backend not yet implemented"

    def get_state(self) -> BackendState:
        return BackendState(status=BackendStatus.STOPPED)

    def list_models(self) -> list[Model]:
        return []

    def supports_cluster(self) -> bool:
        # SGLang supports tensor parallelism but we haven't implemented it
        return False

    def get_cluster_nodes(self) -> list[Node]:
        return [Node(
            id="local",
            hostname="localhost",
            role="local",
            status=NodeStatus.OFFLINE,
        )]
