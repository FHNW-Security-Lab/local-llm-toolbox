"""FastAPI router that proxies to the active backend."""

import asyncio
import logging

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from manager import Backend, get_active_backend
from manager.registry import get_registry

from .config import config

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Router", version="0.1.0")

# Shared async client
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    return _client


# Cache for active backend (refreshed periodically)
_cached_backend: Backend | None = None
_cache_lock = asyncio.Lock()


async def get_cached_backend() -> Backend | None:
    """Get cached active backend."""
    return _cached_backend


async def refresh_backend_cache():
    """Refresh the cached active backend."""
    global _cached_backend
    async with _cache_lock:
        _cached_backend = get_active_backend()


@app.on_event("startup")
async def startup():
    """Start background task to refresh backend cache."""
    await refresh_backend_cache()

    async def refresh_loop():
        while True:
            await asyncio.sleep(config.health_interval)
            await refresh_backend_cache()

    asyncio.create_task(refresh_loop())


@app.on_event("shutdown")
async def shutdown():
    """Clean up client."""
    global _client
    if _client:
        await _client.aclose()
        _client = None


@app.get("/status")
async def status():
    """Return current router status."""
    await refresh_backend_cache()
    backend = await get_cached_backend()
    active_list = get_registry().get_all_active()

    if len(active_list) > 1:
        return JSONResponse(
            {
                "active": None,
                "error": "Multiple backends running",
                "backends": [b.name for b in active_list],
            },
            status_code=503,
        )

    return {
        "active": backend.name if backend else None,
        "display_name": backend.display_name if backend else None,
        "api_base": backend.api_base if backend else None,
    }


async def proxy_request(request: Request, backend: Backend, path: str) -> Response:
    """Proxy a request to the backend."""
    client = get_client()

    # Build target URL
    url = f"{backend.api_base}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    # Get request body
    body = await request.body()

    # Check if streaming requested
    is_streaming = False
    if body and request.method == "POST":
        try:
            import json
            data = json.loads(body)
            is_streaming = data.get("stream", False)
        except (json.JSONDecodeError, KeyError):
            pass

    # Forward headers (excluding host)
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    try:
        if is_streaming:
            # Streaming response
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
            # Regular response
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
        logger.error(f"Proxy error: {e}")
        return JSONResponse(
            {"error": f"Backend request failed: {e}"},
            status_code=502,
        )


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy_v1(request: Request, path: str):
    """Proxy OpenAI-compatible /v1/* endpoints."""
    backend = await get_cached_backend()
    if not backend:
        return JSONResponse(
            {"error": "No backend running"},
            status_code=503,
        )
    return await proxy_request(request, backend, f"v1/{path}")


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy_api(request: Request, path: str):
    """Proxy Ollama-native /api/* endpoints."""
    backend = await get_cached_backend()
    if not backend:
        return JSONResponse(
            {"error": "No backend running"},
            status_code=503,
        )
    return await proxy_request(request, backend, f"api/{path}")
