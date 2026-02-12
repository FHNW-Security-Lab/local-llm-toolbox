"""Base RPC worker control API.

Provides common endpoints that any backend's RPC worker can inherit:
- /health  - simple health check
- /stats   - system stats (memory, GPU)

Backend-specific RPC workers (e.g., llama_rpc.py) extend this with
their own endpoints like /status and /reset.
"""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from ..base import collect_system_stats

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"


class StatsResponse(BaseModel):
    gpu_name: str = ""
    gpu_busy_percent: int = 0
    gpu_memory_used: int = 0
    gpu_memory_total: int = 0
    cpu_percent: float = 0.0
    memory_used: int = 0
    memory_total: int = 0


# ─────────────────────────────────────────────────────────────────
# Base RPC worker server
# ─────────────────────────────────────────────────────────────────


class BaseRpcWorkerServer:
    """Base class for RPC worker control APIs.

    Provides /health and /stats endpoints. Subclasses add
    backend-specific endpoints (e.g., /status, /reset for llama.cpp).
    """

    def __init__(self, app_title: str = "RPC Worker Control API"):
        self._app = FastAPI(title=app_title)
        self._setup_base_routes()

    @property
    def app(self) -> FastAPI:
        return self._app

    def _setup_base_routes(self) -> None:
        """Setup common routes available on all RPC workers."""

        @self._app.get("/health", response_model=HealthResponse)
        async def health() -> HealthResponse:
            """Health check endpoint."""
            return HealthResponse(status="ok")

        @self._app.get("/stats", response_model=StatsResponse)
        async def stats() -> StatsResponse:
            """System stats (memory, GPU) for this worker."""
            data = collect_system_stats()
            logger.debug(f"Stats request: {data}")
            return StatsResponse(**data)
