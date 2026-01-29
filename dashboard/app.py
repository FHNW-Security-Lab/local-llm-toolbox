"""Dashboard + Router unified FastAPI application.

This single app provides both the dashboard UI and the OpenAI-compatible API proxy.
"""

import logging
from dataclasses import asdict

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from manager import (
    get_all_backends,
    get_backend,
    get_active_backend,
    status as get_status,
    start_backend as do_start_backend,
    stop_backend as do_stop_backend,
    list_models,
    download_model,
    load_model,
    unload_model,
    get_backend_state,
)

logger = logging.getLogger("dashboard.app")

app = FastAPI(title="Local LLM Toolbox")

# Shared HTTP client for proxying
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    return _client


@app.on_event("shutdown")
async def shutdown():
    global _client
    if _client:
        await _client.aclose()
        _client = None


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
def api_status() -> StatusResponse:
    """Get current status of all backends."""
    st = get_status()
    backends = get_all_backends()
    active_name = st.active.name if st.active else None

    return StatusResponse(
        active=active_name,
        backends={
            name: {
                "display_name": b.display_name,
                "healthy": st.backends.get(name, False),
                "available": b.is_available(),
                "unavailable_reason": b.get_unavailable_reason(),
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


@app.post("/api/start")
def api_start(req: StartRequest) -> ActionResponse:
    """Start a backend (automatically stops any running backend)."""
    logger.info(f"Starting backend: {req.backend}")
    success, message = do_start_backend(req.backend, force=True)
    if success:
        logger.info(f"Backend started: {req.backend}")
    else:
        logger.warning(f"Failed to start backend {req.backend}: {message}")
    return ActionResponse(success=success, message=message)


@app.post("/api/stop/{name}")
def api_stop(name: str) -> ActionResponse:
    """Stop a specific backend."""
    logger.info(f"Stopping backend: {name}")
    backend = get_backend(name)
    if not backend:
        logger.warning(f"Unknown backend: {name}")
        return ActionResponse(success=False, message=f"Unknown backend: {name}")

    success = do_stop_backend(backend)
    if success:
        logger.info(f"Backend stopped: {name}")
    else:
        logger.warning(f"Failed to stop backend: {name}")
    return ActionResponse(
        success=success,
        message=f"{backend.display_name} stopped" if success else f"Failed to stop {backend.display_name}",
    )


# ─────────────────────────────────────────────────────────────────
# Model Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/api/models")
def api_list_models(backend: str | None = None) -> ModelsResponse:
    """List models for the active backend."""
    models = list_models(backend)

    return ModelsResponse(
        models={
            name: [asdict(m) for m in model_list]
            for name, model_list in models.items()
        },
    )


@app.post("/api/models/load")
def api_load_model(req: LoadModelRequest) -> ActionResponse:
    """Load a model on a backend."""
    logger.info(f"Loading model: {req.model_id} on {req.backend}")
    success, message = load_model(req.backend, req.model_id)
    if success:
        logger.info(f"Model loaded: {req.model_id}")
    else:
        logger.warning(f"Failed to load model {req.model_id}: {message}")
    return ActionResponse(success=success, message=message)


@app.post("/api/models/unload/{backend}")
def api_unload_model(backend: str) -> ActionResponse:
    """Unload the current model from a backend."""
    logger.info(f"Unloading model from {backend}")
    success, message = unload_model(backend)
    if success:
        logger.info(f"Model unloaded from {backend}")
    else:
        logger.warning(f"Failed to unload model: {message}")
    return ActionResponse(success=success, message=message)


@app.post("/api/models/download")
def api_download_model(req: LoadModelRequest) -> ActionResponse:
    """Download a model for a backend."""
    logger.info(f"Downloading model: {req.model_id} for {req.backend}")
    success, message = download_model(req.backend, req.model_id)
    if success:
        logger.info(f"Model downloaded: {req.model_id}")
    else:
        logger.warning(f"Failed to download model {req.model_id}: {message}")
    return ActionResponse(success=success, message=message)


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
            import json
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
    logger.info(f"Chat URL: {chat_url}, model_id: {model_id}")

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
                    yield f"data: {error.decode()}\n\n"
                    return

                async for line in resp.aiter_lines():
                    if line:
                        yield f"{line}\n"
                    else:
                        yield "\n"
        except httpx.RequestError as e:
            yield f'data: {{"error": "Backend request failed: {e}"}}\n\n'

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )


# ─────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the dashboard UI."""
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local LLM Toolbox</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }

        .layout {
            display: grid;
            grid-template-columns: 380px 1fr;
            height: 100vh;
        }

        /* Left Panel - Controls */
        .control-panel {
            background: #0f0f0f;
            border-right: 1px solid #333;
            overflow-y: auto;
            padding: 1.5rem;
        }
        .control-panel h1 {
            font-size: 1.25rem;
            font-weight: 500;
            margin-bottom: 1rem;
            color: #fff;
        }
        .control-panel h2 {
            font-size: 0.9rem;
            font-weight: 500;
            margin: 1.25rem 0 0.75rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .status-bar {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
            font-size: 0.85rem;
        }
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-indicator.active { background: #22c55e; }
        .status-indicator.inactive { background: #666; }

        .cards { display: flex; flex-direction: column; gap: 0.5rem; }

        .card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            transition: border-color 0.2s;
        }
        .card:hover { border-color: #444; }
        .card.active { border-color: #22c55e; }
        .card.unavailable { opacity: 0.5; }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-info h3 { font-size: 0.85rem; font-weight: 500; }
        .card-info .subtitle {
            font-size: 0.7rem;
            color: #666;
            font-family: monospace;
            margin-top: 0.1rem;
        }

        .status-badge {
            display: inline-block;
            font-size: 0.6rem;
            padding: 0.1rem 0.35rem;
            border-radius: 3px;
            margin-left: 0.4rem;
            background: #333;
            color: #888;
        }
        .status-badge.loaded { background: #166534; color: #86efac; }
        .status-badge.online { background: #166534; color: #86efac; }
        .status-badge.offline { background: #7f1d1d; color: #fca5a5; }

        button {
            padding: 0.35rem 0.7rem;
            border-radius: 4px;
            border: none;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-primary { background: #3b82f6; color: #fff; }
        .btn-primary:hover:not(:disabled) { background: #2563eb; }
        .btn-success { background: #22c55e; color: #000; }
        .btn-success:hover:not(:disabled) { background: #16a34a; }
        .btn-danger { background: #ef4444; color: #fff; }
        .btn-danger:hover:not(:disabled) { background: #dc2626; }
        .btn-secondary { background: #333; color: #e0e0e0; }
        .btn-secondary:hover:not(:disabled) { background: #444; }

        .node-stats {
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
            font-size: 0.7rem;
        }
        .stat-item { display: flex; flex-direction: column; gap: 0.1rem; }
        .stat-label { color: #666; font-size: 0.6rem; text-transform: uppercase; }
        .stat-value { color: #e0e0e0; }
        .progress-bar {
            width: 60px;
            height: 3px;
            background: #333;
            border-radius: 2px;
            overflow: hidden;
            margin-top: 2px;
        }
        .progress-fill { height: 100%; background: #3b82f6; transition: width 0.3s; }
        .progress-fill.high { background: #ef4444; }
        .progress-fill.medium { background: #f59e0b; }

        .loading {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid transparent;
            border-top-color: currentColor;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 4px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Available models collapsible */
        .available-models {
            margin-top: 0.5rem;
            border: 1px solid #333;
            border-radius: 8px;
            overflow: hidden;
        }
        .available-models summary {
            padding: 0.75rem 1rem;
            background: #1a1a1a;
            color: #888;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .available-models summary:hover { background: #222; }
        .available-models[open] summary { border-bottom: 1px solid #333; }
        .available-models-list {
            max-height: 300px;
            overflow-y: auto;
            padding: 0.5rem;
        }
        .available-models-list .card {
            opacity: 0.7;
        }
        .available-models-list .card:hover {
            opacity: 1;
        }

        /* Right Panel - Chat */
        .chat-panel {
            display: flex;
            flex-direction: column;
            height: 100vh;
            background: #141414;
        }

        .chat-header {
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #333;
            font-size: 0.9rem;
            color: #888;
        }
        .chat-header strong { color: #fff; }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .message {
            max-width: 85%;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .message.user {
            align-self: flex-end;
            background: #3b82f6;
            color: #fff;
            border-bottom-right-radius: 4px;
        }
        .message.assistant {
            align-self: flex-start;
            background: #1a1a1a;
            border: 1px solid #333;
            border-bottom-left-radius: 4px;
        }
        .message.assistant .content { color: #e0e0e0; }
        .message.error {
            background: #7f1d1d;
            color: #fca5a5;
        }

        .message-stats {
            margin-top: 0.5rem;
            padding-top: 0.4rem;
            border-top: 1px solid #333;
            font-size: 0.7rem;
            color: #666;
        }

        .chat-input-area {
            padding: 1rem 1.5rem;
            border-top: 1px solid #333;
            background: #0f0f0f;
        }
        .chat-input-wrapper {
            display: flex;
            gap: 0.75rem;
        }
        .chat-input {
            flex: 1;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            color: #e0e0e0;
            font-size: 0.9rem;
            font-family: inherit;
            resize: none;
            min-height: 44px;
            max-height: 150px;
        }
        .chat-input:focus { outline: none; border-color: #3b82f6; }
        .chat-input::placeholder { color: #666; }

        .send-btn {
            padding: 0.75rem 1.5rem;
            background: #3b82f6;
            color: #fff;
            border: none;
            border-radius: 6px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .send-btn:hover:not(:disabled) { background: #2563eb; }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .empty-chat {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 0.9rem;
            text-align: center;
            padding: 2rem;
        }

        .toast {
            position: fixed;
            bottom: 2rem;
            left: 50%;
            transform: translateX(-50%) translateY(1rem);
            padding: 0.8rem 1.2rem;
            border-radius: 6px;
            font-size: 0.85rem;
            opacity: 0;
            transition: all 0.3s;
            z-index: 1000;
        }
        .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
        .toast.success { background: #166534; color: #fff; }
        .toast.error { background: #991b1b; color: #fff; }

        .cursor { animation: blink 1s infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
    </style>
</head>
<body>
    <div class="layout">
        <!-- Left Panel - Controls -->
        <div class="control-panel">
            <h1>Local LLM Toolbox</h1>

            <div class="status-bar" id="status-bar">
                <span class="status-indicator inactive"></span>Loading...
            </div>

            <div id="nodes-section" style="display: none;">
                <h2>Cluster</h2>
                <div class="cards" id="nodes"></div>
            </div>

            <h2>Backends</h2>
            <div class="cards" id="backends"></div>

            <div id="models-section" style="display: none;">
                <h2>Models</h2>
                <div class="cards" id="models"></div>
            </div>
        </div>

        <!-- Right Panel - Chat -->
        <div class="chat-panel">
            <div class="chat-header" id="chat-header">
                <strong>Chat</strong> - No model loaded
            </div>
            <div class="chat-messages" id="chat-messages">
                <div class="empty-chat" id="empty-chat">
                    Load a model to start chatting
                </div>
            </div>
            <div class="chat-input-area">
                <div class="chat-input-wrapper">
                    <textarea
                        class="chat-input"
                        id="chat-input"
                        placeholder="Type a message..."
                        rows="1"
                        disabled
                    ></textarea>
                    <button class="send-btn" id="send-btn" disabled>Send</button>
                </div>
            </div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        let state = { status: null, models: {}, backendState: null, loading: {} };
        let chatMessages = [];
        let isGenerating = false;
        let abortController = null;
        let currentModelId = null;

        async function fetchAll() {
            try {
                const statusResp = await fetch('/api/status');
                state.status = await statusResp.json();

                // Only fetch models if a backend is active
                if (state.status?.active) {
                    const [modelsResp, stateResp] = await Promise.all([
                        fetch('/api/models'),
                        fetch(`/api/backends/${state.status.active}`)
                    ]);
                    state.models = (await modelsResp.json()).models;
                    state.backendState = await stateResp.json();
                } else {
                    state.models = {};
                    state.backendState = null;
                }

                render();
            } catch (e) {
                console.error('Failed to fetch:', e);
            }
        }

        function formatSize(bytes) {
            if (!bytes) return '';
            const gb = bytes / (1024 * 1024 * 1024);
            return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
        }

        function render() {
            renderStatus();
            renderNodes();
            renderBackends();
            renderModels();
            renderChatHeader();
            updateChatInput();
        }

        function renderStatus() {
            const el = document.getElementById('status-bar');
            if (state.status?.active) {
                const backend = state.status.backends[state.status.active];
                let text = `<span class="status-indicator active"></span>${backend.display_name}`;
                if (state.backendState?.loaded_model) {
                    text += ` - ${state.backendState.loaded_model.name}`;
                }
                el.innerHTML = text;
            } else {
                el.innerHTML = '<span class="status-indicator inactive"></span>No backend running';
            }
        }

        function renderNodes() {
            const section = document.getElementById('nodes-section');
            const el = document.getElementById('nodes');

            if (!state.backendState?.nodes?.length) {
                section.style.display = 'none';
                return;
            }

            section.style.display = 'block';
            el.innerHTML = state.backendState.nodes.map(node => {
                const isOnline = node.status === 'online';
                const badge = isOnline
                    ? '<span class="status-badge online">online</span>'
                    : '<span class="status-badge offline">offline</span>';

                const memPct = node.memory_total ? Math.round(node.memory_used / node.memory_total * 100) : 0;
                const gpuPct = node.gpu_memory_total ? Math.round(node.gpu_memory_used / node.gpu_memory_total * 100) : 0;
                const memClass = memPct > 90 ? 'high' : memPct > 70 ? 'medium' : '';
                const gpuClass = gpuPct > 90 ? 'high' : gpuPct > 70 ? 'medium' : '';

                return `
                    <div class="card ${isOnline ? 'active' : ''}">
                        <div class="card-header">
                            <div class="card-info">
                                <h3>${node.hostname}${badge}</h3>
                                <div class="subtitle">${node.role} · ${node.gpu_name || 'CPU'}</div>
                            </div>
                        </div>
                        ${isOnline && node.memory_total ? `
                        <div class="node-stats">
                            <div class="stat-item">
                                <span class="stat-label">Mem</span>
                                <span class="stat-value">${formatSize(node.memory_used)} / ${formatSize(node.memory_total)}</span>
                                <div class="progress-bar"><div class="progress-fill ${memClass}" style="width: ${memPct}%"></div></div>
                            </div>
                            ${node.gpu_memory_total ? `
                            <div class="stat-item">
                                <span class="stat-label">GPU</span>
                                <span class="stat-value">${formatSize(node.gpu_memory_used)} / ${formatSize(node.gpu_memory_total)}</span>
                                <div class="progress-bar"><div class="progress-fill ${gpuClass}" style="width: ${gpuPct}%"></div></div>
                            </div>
                            ` : ''}
                        </div>
                        ` : ''}
                    </div>
                `;
            }).join('');
        }

        function renderBackends() {
            const el = document.getElementById('backends');
            if (!state.status) return;

            el.innerHTML = Object.entries(state.status.backends).map(([name, b]) => {
                const isActive = state.status.active === name;
                const loading = state.loading[`backend-${name}`];
                let badge = b.unavailable_reason ? `<span class="status-badge">${b.unavailable_reason}</span>` : '';

                let btn = '';
                if (isActive) {
                    btn = `<button class="btn-danger" onclick="stopBackend('${name}')" ${loading ? 'disabled' : ''}>
                        ${loading ? '<span class="loading"></span>' : ''}Stop</button>`;
                } else if (b.available) {
                    btn = `<button class="btn-success" onclick="startBackend('${name}')" ${loading ? 'disabled' : ''}>
                        ${loading ? '<span class="loading"></span>' : ''}${state.status.active ? 'Switch' : 'Start'}</button>`;
                }

                return `
                    <div class="card ${isActive ? 'active' : ''} ${!b.available ? 'unavailable' : ''}">
                        <div class="card-header">
                            <div class="card-info">
                                <h3>${b.display_name}${badge}</h3>
                                <div class="subtitle">${b.model_format || 'unknown'}</div>
                            </div>
                            <div>${btn}</div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function renderModels() {
            const section = document.getElementById('models-section');
            const el = document.getElementById('models');

            if (!state.status?.active) {
                section.style.display = 'none';
                return;
            }

            section.style.display = 'block';
            const activeBackend = state.status.active;
            const allModels = state.models[activeBackend] || [];
            const loadedId = state.backendState?.loaded_model?.id;

            if (allModels.length === 0) {
                el.innerHTML = '<div style="color:#666;font-size:0.8rem;padding:0.5rem;">No models found</div>';
                return;
            }

            // Split into downloaded and available for download
            const downloaded = allModels.filter(m => m.downloaded !== false);
            const available = allModels.filter(m => m.downloaded === false);

            // Sort downloaded: loaded first
            downloaded.sort((a, b) => (b.id === loadedId ? 1 : 0) - (a.id === loadedId ? 1 : 0));

            function renderModelCard(m, isAvailable = false) {
                const loading = state.loading[`model-${m.id}`];
                const downloading = state.loading[`download-${m.id}`];
                const isLoaded = m.id === loadedId;

                let btn = '';
                if (isLoaded) {
                    btn = `<button class="btn-secondary" onclick="unloadModel('${activeBackend}')" ${loading ? 'disabled' : ''}>Unload</button>`;
                } else if (!isAvailable) {
                    btn = `<button class="btn-primary" onclick="loadModel('${activeBackend}', '${m.id}')" ${loading ? 'disabled' : ''}>${loading ? '<span class="loading"></span>' : ''}${loadedId ? 'Switch' : 'Load'}</button>`;
                } else {
                    btn = `<button class="btn-success" onclick="downloadModel('${activeBackend}', '${m.id}')" ${downloading ? 'disabled' : ''}>${downloading ? '<span class="loading"></span>' : ''}Download</button>`;
                }

                return `
                    <div class="card ${isLoaded ? 'active' : ''}">
                        <div class="card-header">
                            <div class="card-info">
                                <h3>${m.name}${m.quantization ? ` <span class="status-badge">${m.quantization}</span>` : ''}${isLoaded ? ' <span class="status-badge loaded">loaded</span>' : ''}</h3>
                                <div class="subtitle">${formatSize(m.size_bytes)}</div>
                            </div>
                            <div>${btn}</div>
                        </div>
                    </div>
                `;
            }

            let html = downloaded.map(m => renderModelCard(m, false)).join('');

            // Add collapsible section for available models
            if (available.length > 0) {
                html += `
                    <details class="available-models">
                        <summary>Available for download (${available.length})</summary>
                        <div class="available-models-list">
                            ${available.map(m => renderModelCard(m, true)).join('')}
                        </div>
                    </details>
                `;
            }

            el.innerHTML = html;
        }

        function renderChatHeader() {
            const el = document.getElementById('chat-header');
            const newModelId = state.backendState?.loaded_model?.id || null;

            // Clear chat if model changed
            if (newModelId !== currentModelId) {
                chatMessages = [];
                currentModelId = newModelId;
                renderMessages();
            }

            if (state.backendState?.loaded_model) {
                el.innerHTML = `<strong>Chat</strong> - ${state.backendState.loaded_model.name}`;
            } else {
                el.innerHTML = '<strong>Chat</strong> - No model loaded';
            }
        }

        function updateChatInput() {
            const input = document.getElementById('chat-input');
            const btn = document.getElementById('send-btn');
            const hasModel = !!state.backendState?.loaded_model;

            input.disabled = !hasModel || isGenerating;
            btn.disabled = !hasModel || isGenerating;
            input.placeholder = hasModel ? 'Type a message...' : 'Load a model to start chatting';

            const emptyChat = document.getElementById('empty-chat');
            if (emptyChat) {
                emptyChat.style.display = chatMessages.length === 0 ? 'flex' : 'none';
                emptyChat.textContent = hasModel ? 'Start a conversation' : 'Load a model to start chatting';
            }
        }

        function renderMessages() {
            const container = document.getElementById('chat-messages');
            const emptyEl = document.getElementById('empty-chat');

            if (chatMessages.length === 0) {
                container.innerHTML = `<div class="empty-chat" id="empty-chat">${state.backendState?.loaded_model ? 'Start a conversation' : 'Load a model to start chatting'}</div>`;
                return;
            }

            container.innerHTML = chatMessages.map((msg, i) => {
                if (msg.role === 'user') {
                    return `<div class="message user">${escapeHtml(msg.content)}</div>`;
                } else {
                    let statsHtml = '';
                    if (msg.stats && !msg.generating) {
                        const s = msg.stats;
                        const parts = [];
                        if (s.tokens_per_sec) parts.push(`${s.tokens_per_sec.toFixed(1)} tok/s`);
                        if (s.time_ms) parts.push(`${(s.time_ms / 1000).toFixed(1)}s`);
                        if (s.prompt_tokens && s.completion_tokens) parts.push(`${s.prompt_tokens}→${s.completion_tokens} tokens`);
                        if (parts.length) {
                            statsHtml = `<div class="message-stats">${parts.join(' · ')}</div>`;
                        }
                    }
                    const cursor = msg.generating ? '<span class="cursor">▌</span>' : '';
                    return `<div class="message assistant"><div class="content">${escapeHtml(msg.content)}${cursor}</div>${statsHtml}</div>`;
                }
            }).join('');

            container.scrollTop = container.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const text = input.value.trim();
            if (!text || isGenerating) return;

            input.value = '';
            input.style.height = 'auto';

            chatMessages.push({ role: 'user', content: text });
            chatMessages.push({ role: 'assistant', content: '', generating: true });
            renderMessages();

            isGenerating = true;
            updateChatInput();

            const msgIndex = chatMessages.length - 1;
            abortController = new AbortController();

            try {
                const messages = chatMessages.slice(0, -1).map(m => ({
                    role: m.role,
                    content: m.content
                }));

                const resp = await fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages }),
                    signal: abortController.signal
                });

                if (!resp.ok) {
                    throw new Error(`HTTP ${resp.status}`);
                }

                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let finalStats = null;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const data = line.slice(6);
                        if (data === '[DONE]') continue;

                        try {
                            const chunk = JSON.parse(data);

                            if (chunk.choices?.[0]?.delta?.content) {
                                chatMessages[msgIndex].content += chunk.choices[0].delta.content;
                                renderMessages();
                            }

                            // llama.cpp sends timings in final chunk
                            if (chunk.timings) {
                                finalStats = finalStats || {};
                                finalStats.tokens_per_sec = chunk.timings.predicted_per_second;
                                finalStats.time_ms = chunk.timings.predicted_ms;
                                finalStats.prompt_tokens = chunk.timings.prompt_n;
                                finalStats.completion_tokens = chunk.timings.predicted_n;
                            }

                            // OpenAI-style usage (fallback)
                            if (chunk.usage) {
                                finalStats = finalStats || {};
                                finalStats.prompt_tokens = finalStats.prompt_tokens || chunk.usage.prompt_tokens;
                                finalStats.completion_tokens = finalStats.completion_tokens || chunk.usage.completion_tokens;
                            }
                        } catch (e) {}
                    }
                }

                chatMessages[msgIndex].generating = false;
                if (finalStats) {
                    chatMessages[msgIndex].stats = finalStats;
                }
                renderMessages();

            } catch (e) {
                if (e.name !== 'AbortError') {
                    chatMessages[msgIndex].content = `Error: ${e.message}`;
                    chatMessages[msgIndex].generating = false;
                }
                renderMessages();
            } finally {
                isGenerating = false;
                abortController = null;
                updateChatInput();
            }
        }

        // Event listeners
        document.getElementById('chat-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        document.getElementById('chat-input').addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
        });

        document.getElementById('send-btn').addEventListener('click', sendMessage);

        async function startBackend(name) {
            state.loading[`backend-${name}`] = true;
            render();
            try {
                const resp = await fetch('/api/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ backend: name })
                });
                const result = await resp.json();
                showToast(result.message, result.success ? 'success' : 'error');
                await fetchAll();
            } finally {
                state.loading[`backend-${name}`] = false;
                render();
            }
        }

        async function stopBackend(name) {
            state.loading[`backend-${name}`] = true;
            render();
            try {
                const resp = await fetch(`/api/stop/${name}`, { method: 'POST' });
                const result = await resp.json();
                showToast(result.message, result.success ? 'success' : 'error');
                await fetchAll();
            } finally {
                state.loading[`backend-${name}`] = false;
                render();
            }
        }

        async function loadModel(backend, modelId) {
            state.loading[`model-${modelId}`] = true;
            render();
            try {
                const resp = await fetch('/api/models/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ backend, model_id: modelId })
                });
                const result = await resp.json();
                showToast(result.message, result.success ? 'success' : 'error');
                await fetchAll();
            } finally {
                state.loading[`model-${modelId}`] = false;
                render();
            }
        }

        async function unloadModel(backend) {
            state.loading['unload'] = true;
            render();
            try {
                const resp = await fetch(`/api/models/unload/${backend}`, { method: 'POST' });
                const result = await resp.json();
                showToast(result.message, result.success ? 'success' : 'error');
                await fetchAll();
            } finally {
                state.loading['unload'] = false;
                render();
            }
        }

        async function downloadModel(backend, modelId) {
            state.loading[`download-${modelId}`] = true;
            render();
            try {
                const resp = await fetch('/api/models/download', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ backend, model_id: modelId })
                });
                const result = await resp.json();
                showToast(result.message, result.success ? 'success' : 'error');
                if (result.success) {
                    await fetchAll();
                }
            } finally {
                state.loading[`download-${modelId}`] = false;
                render();
            }
        }

        function showToast(message, type) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = `toast ${type} show`;
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        fetchAll();
        setInterval(fetchAll, 10000);
    </script>
</body>
</html>
"""
