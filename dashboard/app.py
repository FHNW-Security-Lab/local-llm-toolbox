"""Dashboard + Router unified FastAPI application.

This single app provides both the dashboard UI and the OpenAI-compatible API proxy.
Uses manager/bridge.py for async-to-sync bridging.
"""

import asyncio
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from manager import (
    get_all_backends,
    get_backend,
    get_active_backend,
    get_backend_state,
)
from manager.registry import get_registry
from manager.bridge import (
    async_start_backend,
    async_stop_backend,
    async_status,
    async_list_models,
    async_load_model,
    async_unload_model,
    shutdown as bridge_shutdown,
)
from manager.tasks import get_task_manager, Task, TaskStatus

logger = logging.getLogger("dashboard.app")

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


# ─────────────────────────────────────────────────────────────────
# Application Lifespan
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown").
    """
    logger.info("Dashboard starting up")

    # Register task manager callback for SSE broadcasts
    def on_task_update(task: Task):
        broadcast_task_event(task)

    get_task_manager().on_task_update(on_task_update)
    logger.debug("Registered task manager callback")

    # Initialize shutdown event for this event loop
    shutdown_event = get_shutdown_event()

    try:
        yield  # Application runs here
    except asyncio.CancelledError:
        logger.debug("Lifespan cancelled (shutdown signal)")
    finally:
        # Signal all SSE generators to stop
        logger.debug("Setting shutdown event")
        shutdown_event.set()

        # Give SSE generators a moment to clean up
        await asyncio.sleep(0.1)

        # Shutdown - runs even if cancelled
        logger.info("Dashboard shutting down")
        global _client
        if _client:
            try:
                await _client.aclose()
            except Exception as e:
                logger.debug(f"Error closing HTTP client: {e}")
            _client = None
        bridge_shutdown()
        logger.info("Dashboard shutdown complete")


app = FastAPI(title="Local LLM Toolbox", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────────────────────────────────────────────
# Shutdown Coordination
# ─────────────────────────────────────────────────────────────────

# Event to signal SSE generators to stop
_shutdown_event: asyncio.Event | None = None


def get_shutdown_event() -> asyncio.Event:
    """Get or create the shutdown event for the current event loop."""
    global _shutdown_event
    if _shutdown_event is None:
        _shutdown_event = asyncio.Event()
    return _shutdown_event


# ─────────────────────────────────────────────────────────────────
# Operation Event Broadcasting
# ─────────────────────────────────────────────────────────────────

# Operation events queue for SSE clients
_operation_events: deque = deque(maxlen=100)
_event_counter = 0
_state_refresh_needed = False  # Flag to trigger immediate state refresh


def broadcast_operation(
    operation: str,
    status: str,
    backend: str | None = None,
    model: str | None = None,
    message: str = "",
    error: str | None = None,
):
    """Broadcast an operation event to all SSE clients."""
    global _event_counter, _state_refresh_needed
    _event_counter += 1
    event = {
        "type": "operation",
        "id": _event_counter,
        "operation": operation,
        "status": status,
        "backend": backend,
        "model": model,
        "message": message,
        "error": error,
        "timestamp": time.time(),
    }
    _operation_events.append(event)
    logger.debug(f"SSE broadcast: {operation} {status} - {message or error or ''}")

    # Trigger immediate state refresh when operations complete/fail
    if status in ("completed", "failed"):
        _state_refresh_needed = True


def broadcast_task_event(task: Task):
    """Broadcast a task update as an SSE event."""
    status_map = {
        TaskStatus.PENDING: "pending",
        TaskStatus.RUNNING: "started",
        TaskStatus.COMPLETED: "completed",
        TaskStatus.FAILED: "failed",
        TaskStatus.CANCELLED: "cancelled",
    }
    broadcast_operation(
        operation=f"task:{task.operation}",
        status=status_map.get(task.status, "unknown"),
        backend=task.backend,
        model=task.params.get("model_id"),
        message=task.message,
        error=task.error,
    )


# Shared HTTP client for proxying
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    return _client


# ─────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    backend: str


class LoadModelRequest(BaseModel):
    backend: str
    model_id: str


class StatusResponse(BaseModel):
    active: str | None
    backends: dict[str, dict]


class ActionResponse(BaseModel):
    success: bool
    message: str
    task_id: str | None = None


class ChatRequest(BaseModel):
    messages: list[dict]


class ModelsResponse(BaseModel):
    models: dict[str, list[dict]]


class BackendStateResponse(BaseModel):
    status: str
    model_status: str
    loaded_model: dict | None
    nodes: list[dict]
    error: str | None


# ─────────────────────────────────────────────────────────────────
# Backend Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status() -> StatusResponse:
    """Get current status of all backends."""
    logger.debug("API: GET /api/status")
    st = await async_status()
    backends = get_all_backends()
    registry = get_registry()
    active_name = st.active.name if st.active else None

    return StatusResponse(
        active=active_name,
        backends={
            name: {
                "display_name": b.display_name,
                "healthy": st.backends.get(name, False),
                # Use cached availability from registry (checked once at startup)
                "available": registry.is_available(name),
                "unavailable_reason": registry.get_unavailable_reason(name),
                # Only get api_base for active backend (Foundry SDK init is slow)
                "api_base": b.api_base if name == active_name else "",
                "description": b.description,
                "model_format": b.model_format,
                "supports_cluster": b.supports_cluster(),
            }
            for name, b in backends.items()
        },
    )


@app.get("/api/backends/{name}")
def api_backend_state(name: str) -> BackendStateResponse:
    """Get detailed state of a specific backend."""
    logger.debug(f"API: GET /api/backends/{name}")
    state = get_backend_state(name)
    if not state:
        return BackendStateResponse(
            status="unknown",
            model_status="unknown",
            loaded_model=None,
            nodes=[],
            error=f"Backend not found: {name}",
        )

    return BackendStateResponse(
        status=state.status.value,
        model_status=state.model_status.value,
        loaded_model=asdict(state.loaded_model) if state.loaded_model else None,
        nodes=[asdict(n) for n in state.nodes],
        error=state.error,
    )


@app.get("/api/events")
async def api_events(request: Request):
    """Server-Sent Events stream for real-time state updates.

    The frontend connects to this endpoint and receives:
    - "state" events: periodic state snapshots when state changes
    - "operation" events: started/completed/failed for long-running operations
    - "task:*" events: task system updates

    Uses sse-starlette for proper shutdown handling.
    """
    logger.debug("API: SSE client connected to /api/events")
    shutdown_event = get_shutdown_event()

    async def event_generator():
        global _state_refresh_needed
        last_state_json = None
        last_event_id = 0
        last_state_poll = 0
        state_poll_interval = 10.0  # Poll backend state every 10 seconds

        while not shutdown_event.is_set():
            # Check for client disconnect
            if await request.is_disconnected():
                logger.debug("SSE: Client disconnected")
                break

            # Check for new operation events (cheap - just reading a deque)
            for event in list(_operation_events):
                if event["id"] > last_event_id:
                    last_event_id = event["id"]
                    yield {"data": json.dumps(event)}

            # Poll backend state:
            # - Every 10s as heartbeat
            # - Immediately when an operation completes (flag set by broadcast_operation)
            now = time.time()
            needs_poll = (now - last_state_poll >= state_poll_interval) or _state_refresh_needed
            if needs_poll:
                if _state_refresh_needed:
                    logger.debug("SSE: Immediate state refresh triggered by operation completion")
                _state_refresh_needed = False
                last_state_poll = now

                # Build current state
                st = await async_status()
                backends = get_all_backends()
                registry = get_registry()
                active_name = st.active.name if st.active else None

                current_state = {
                    "type": "state",
                    "active": active_name,
                    "backends": {
                        name: {
                            "display_name": b.display_name,
                            "healthy": st.backends.get(name, False),
                            # Use cached availability (checked once at startup)
                            "available": registry.is_available(name),
                            "unavailable_reason": registry.get_unavailable_reason(name),
                            "api_base": b.api_base if name == active_name else "",
                            "description": b.description,
                            "model_format": b.model_format,
                            "supports_cluster": b.supports_cluster(),
                        }
                        for name, b in backends.items()
                    },
                }

                # Add backend-specific state if active
                if active_name:
                    backend_state = get_backend_state(active_name)
                    if backend_state:
                        loaded_model_id = backend_state.loaded_model.id if backend_state.loaded_model else None
                        logger.debug(f"SSE: Backend state - loaded_model={loaded_model_id}")
                        current_state["backendState"] = {
                            "status": backend_state.status.value,
                            "model_status": backend_state.model_status.value,
                            "loaded_model": asdict(backend_state.loaded_model) if backend_state.loaded_model else None,
                            "nodes": [asdict(n) for n in backend_state.nodes],
                            "error": backend_state.error,
                        }
                else:
                    current_state["backendState"] = None

                # Only send if state changed
                state_json = json.dumps(current_state, sort_keys=True)
                if state_json != last_state_json:
                    logger.debug("SSE: Sending state update (changed)")
                    last_state_json = state_json
                    yield {"data": state_json}

            # Short sleep to check for events frequently
            await asyncio.sleep(0.1)

        logger.debug("SSE: Generator exiting")

    return EventSourceResponse(event_generator())


@app.post("/api/start")
async def api_start(req: StartRequest, background_tasks: BackgroundTasks) -> ActionResponse:
    """Start a backend.

    This operation runs in the background. Progress is reported via SSE.
    """
    logger.info(f"API: POST /api/start backend={req.backend}")
    registry = get_registry()

    # Quick validation using cached availability
    backend = get_backend(req.backend)
    if not backend:
        logger.warning(f"Unknown backend: {req.backend}")
        return ActionResponse(success=False, message=f"Unknown backend: {req.backend}")
    if not registry.is_available(req.backend):
        reason = registry.get_unavailable_reason(req.backend) or "Not available"
        logger.warning(f"Backend {req.backend} not available: {reason}")
        return ActionResponse(success=False, message=reason)

    # Check if operation already in progress
    if get_task_manager().is_operation_in_progress():
        logger.warning("Cannot start backend: another operation in progress")
        return ActionResponse(success=False, message="Another operation is already in progress")

    # Run in background
    background_tasks.add_task(start_backend_task, req.backend)
    return ActionResponse(success=True, message=f"Starting {backend.display_name}...")


async def start_backend_task(backend_name: str):
    """Background task to start a backend."""
    backend = get_backend(backend_name)
    display_name = backend.display_name if backend else backend_name

    broadcast_operation(
        "backend:start", "started",
        backend=backend_name,
        message=f"Starting {display_name}...",
    )

    try:
        logger.info(f"Starting backend: {backend_name}")
        success, message = await async_start_backend(backend_name, force=True)

        if not success:
            logger.warning(f"Backend start failed: {message}")
            broadcast_operation(
                "backend:start", "failed",
                backend=backend_name,
                error=message,
            )
            return

        # Backend started successfully. Don't call is_healthy() here:
        # - For llama: server isn't running yet (starts when model loads)
        # - For Foundry: would trigger slow SDK init
        # The user needs to load a model next anyway.
        logger.info(f"Backend {backend_name} activated - load a model to begin")
        broadcast_operation(
            "backend:start", "completed",
            backend=backend_name,
            message=f"{display_name} ready - load a model to begin",
        )
    except Exception as e:
        logger.exception(f"Error in start_backend_task: {e}")
        broadcast_operation(
            "backend:start", "failed",
            backend=backend_name,
            error=str(e),
        )


@app.post("/api/stop/{name}")
async def api_stop(name: str, background_tasks: BackgroundTasks) -> ActionResponse:
    """Stop a backend."""
    logger.info(f"API: POST /api/stop/{name}")
    backend = get_backend(name)
    if not backend:
        return ActionResponse(success=False, message=f"Unknown backend: {name}")

    background_tasks.add_task(stop_backend_task, name)
    return ActionResponse(success=True, message=f"Stopping {backend.display_name}...")


async def stop_backend_task(backend_name: str):
    """Background task to stop a backend."""
    backend = get_backend(backend_name)
    display_name = backend.display_name if backend else backend_name

    broadcast_operation(
        "backend:stop", "started",
        backend=backend_name,
        message=f"Stopping {display_name}...",
    )

    try:
        logger.info(f"Stopping backend: {backend_name}")
        success = await async_stop_backend(backend)
        if success:
            logger.info(f"Backend {backend_name} stopped")
            broadcast_operation(
                "backend:stop", "completed",
                backend=backend_name,
                message=f"{display_name} stopped",
            )
        else:
            logger.warning(f"Failed to stop {backend_name}")
            broadcast_operation(
                "backend:stop", "failed",
                backend=backend_name,
                error=f"Failed to stop {display_name}",
            )
    except Exception as e:
        logger.exception(f"Error in stop_backend_task: {e}")
        broadcast_operation(
            "backend:stop", "failed",
            backend=backend_name,
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────────
# Model Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def api_list_models(backend: str | None = None) -> ModelsResponse:
    """List models for the active backend."""
    logger.debug(f"API: GET /api/models backend={backend}")
    models = await async_list_models(backend)

    return ModelsResponse(
        models={
            name: [asdict(m) for m in model_list]
            for name, model_list in models.items()
        },
    )


@app.post("/api/models/load")
async def api_load_model(req: LoadModelRequest) -> ActionResponse:
    """Load a model.

    This is a long-running operation. Returns a task ID immediately.
    Progress is reported via SSE using the task system.
    """
    logger.info(f"API: POST /api/models/load backend={req.backend} model_id={req.model_id}")

    backend = get_backend(req.backend)
    if not backend:
        return ActionResponse(success=False, message=f"Unknown backend: {req.backend}")

    # Submit to task manager (which handles locking)
    task = await async_load_model(req.backend, req.model_id)

    if task.status == TaskStatus.FAILED:
        # Task was rejected (e.g., operation already in progress)
        logger.warning(f"Model load rejected: {task.error}")
        return ActionResponse(success=False, message=task.error or "Operation rejected")

    logger.info(f"Model load task submitted: {task.id}")
    return ActionResponse(
        success=True,
        message=f"Loading {req.model_id}...",
        task_id=task.id,
    )


@app.post("/api/models/unload/{backend}")
async def api_unload_model(backend: str, background_tasks: BackgroundTasks) -> ActionResponse:
    """Unload model."""
    logger.info(f"API: POST /api/models/unload/{backend}")

    backend_obj = get_backend(backend)
    if not backend_obj:
        return ActionResponse(success=False, message=f"Unknown backend: {backend}")

    background_tasks.add_task(unload_model_task, backend)
    return ActionResponse(success=True, message="Unloading model...")


async def unload_model_task(backend_name: str):
    """Background task to unload a model."""
    broadcast_operation(
        "model:unload", "started",
        backend=backend_name,
        message="Unloading model...",
    )

    try:
        logger.info(f"Unloading model from {backend_name}")
        success, message = await async_unload_model(backend_name)
        if success:
            logger.info(f"Model unloaded from {backend_name}")
            broadcast_operation(
                "model:unload", "completed",
                backend=backend_name,
                message="Model unloaded",
            )
        else:
            logger.warning(f"Failed to unload model: {message}")
            broadcast_operation(
                "model:unload", "failed",
                backend=backend_name,
                error=message,
            )
    except Exception as e:
        logger.exception(f"Error in unload_model_task: {e}")
        broadcast_operation(
            "model:unload", "failed",
            backend=backend_name,
            error=str(e),
        )


@app.post("/api/models/download")
async def api_download_model(req: LoadModelRequest, background_tasks: BackgroundTasks) -> ActionResponse:
    """Download a model for a backend.

    This is a long-running operation that runs in the background.
    Progress is reported via SSE.
    """
    logger.info(f"API: POST /api/models/download backend={req.backend} model_id={req.model_id}")

    backend = get_backend(req.backend)
    if not backend:
        return ActionResponse(success=False, message=f"Unknown backend: {req.backend}")

    background_tasks.add_task(download_model_task, req.backend, req.model_id)
    return ActionResponse(success=True, message=f"Downloading {req.model_id}...")


async def download_model_task(backend_name: str, model_id: str):
    """Background task to download a model."""
    broadcast_operation(
        "model:download", "started",
        backend=backend_name,
        model=model_id,
        message=f"Downloading {model_id}...",
    )

    try:
        logger.info(f"Downloading model {model_id} for {backend_name}")
        # Run sync download in executor to not block event loop
        from manager import download_model
        from manager.bridge import run_sync
        success, message = await run_sync(download_model, backend_name, model_id)

        if success:
            logger.info(f"Model downloaded: {model_id}")
            broadcast_operation(
                "model:download", "completed",
                backend=backend_name,
                model=model_id,
                message=f"Downloaded {model_id}",
            )
        else:
            logger.warning(f"Failed to download model {model_id}: {message}")
            broadcast_operation(
                "model:download", "failed",
                backend=backend_name,
                model=model_id,
                error=message,
            )
    except Exception as e:
        logger.exception(f"Error in download_model_task: {e}")
        broadcast_operation(
            "model:download", "failed",
            backend=backend_name,
            model=model_id,
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────────
# Task Status Endpoint
# ─────────────────────────────────────────────────────────────────

@app.get("/api/tasks/{task_id}")
def api_get_task(task_id: str):
    """Get the status of a task."""
    logger.debug(f"API: GET /api/tasks/{task_id}")
    task = get_task_manager().get_task(task_id)
    if not task:
        return JSONResponse({"error": f"Task not found: {task_id}"}, status_code=404)

    return {
        "id": task.id,
        "operation": task.operation,
        "backend": task.backend,
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message,
        "error": task.error,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
    }


# ─────────────────────────────────────────────────────────────────
# OpenAI-Compatible API Proxy (replaces separate router)
# ─────────────────────────────────────────────────────────────────

async def proxy_to_backend(request: Request, path: str) -> Response:
    """Proxy a request to the active backend."""
    backend = get_active_backend()
    if not backend:
        logger.debug(f"Proxy request to /{path} but no active backend")
        return JSONResponse({"error": "No backend running"}, status_code=503)

    api_base = backend.api_base
    if not api_base:
        logger.warning(f"Proxy request but backend {backend.name} has no API base URL")
        return JSONResponse({"error": f"Backend {backend.name} has no API base URL"}, status_code=503)

    client = get_client()
    url = f"{api_base}/{path}"
    logger.debug(f"Proxying {request.method} /{path} -> {url}")
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()

    # Check if streaming
    is_streaming = False
    if body and request.method == "POST":
        try:
            data = json.loads(body)
            is_streaming = data.get("stream", False)
        except (json.JSONDecodeError, KeyError):
            pass

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    try:
        if is_streaming:
            req = client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            )
            resp = await client.send(req, stream=True)

            async def stream_generator():
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                finally:
                    await resp.aclose()

            return StreamingResponse(
                stream_generator(),
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "text/event-stream"),
            )
        else:
            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
    except httpx.RequestError as e:
        logger.error(f"Backend proxy request failed: {e}")
        return JSONResponse({"error": f"Backend request failed: {e}"}, status_code=502)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy_v1(request: Request, path: str):
    """Proxy OpenAI-compatible /v1/* endpoints to active backend."""
    return await proxy_to_backend(request, f"v1/{path}")


@app.api_route("/api/ollama/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy_ollama(request: Request, path: str):
    """Proxy Ollama-native /api/* endpoints."""
    return await proxy_to_backend(request, f"api/{path}")


# ─────────────────────────────────────────────────────────────────
# Chat Endpoint (uses internal proxy)
# ─────────────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest):
    """Stream chat completion from the active backend."""
    backend = get_active_backend()
    if not backend:
        logger.warning("Chat request with no active backend")

        async def error_stream():
            yield 'data: {"error": "No backend running"}\n\n'
        return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=503)

    api_base = backend.api_base
    if not api_base:
        logger.warning(f"Chat request but backend {backend.name} has no API base URL")

        async def error_stream():
            yield f'data: {{"error": "Backend {backend.name} has no API base URL"}}\n\n'
        return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=503)

    # Log the chat request (truncate long messages)
    last_msg = req.messages[-1] if req.messages else {}
    content_preview = str(last_msg.get("content", ""))[:50]
    logger.info(f"Chat request: {len(req.messages)} messages, last: \"{content_preview}...\"")

    # Get the loaded model - required by some backends (e.g., Foundry)
    loaded_model = backend.get_loaded_model()
    model_id = loaded_model.id if loaded_model else None

    # Build the full URL for chat completions
    chat_url = f"{api_base}/v1/chat/completions"
    logger.debug(f"Chat URL: {chat_url}, model_id: {model_id}")

    client = get_client()

    async def stream_response():
        try:
            payload = {
                "messages": req.messages,
                "stream": True,
            }
            # Include model ID if available (required by Foundry)
            if model_id:
                payload["model"] = model_id

            logger.debug(f"Chat payload: model={model_id}, messages={len(req.messages)}")

            async with client.stream(
                "POST",
                chat_url,
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    error = await resp.aread()
                    logger.warning(f"Chat response error: {resp.status_code} - {error.decode()[:200]}")
                    yield f"data: {error.decode()}\n\n"
                    return

                async for line in resp.aiter_lines():
                    if line:
                        yield f"{line}\n"
                    else:
                        yield "\n"
        except httpx.RequestError as e:
            logger.error(f"Chat request failed: {e}")
            yield f'data: {{"error": "Backend request failed: {e}"}}\n\n'

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )


# ─────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    """Serve the dashboard UI."""
    return FileResponse(STATIC_DIR / "index.html")
