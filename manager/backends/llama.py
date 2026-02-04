"""llama.cpp backend implementation.

This backend manages a local llama-server process with optional RPC clustering
for distributed inference. All operations are synchronous - async bridging
is handled by manager/bridge.py.
"""

import asyncio
import logging
import shutil
import socket
import subprocess
import time

from ..base import BaseBackend, has_metal
from ..config import LlamaSettings, get_llama_settings
from ..interface import (
    BackendState,
    BackendStatus,
    ClusterConfig,
    Model,
    ModelStatus,
    Node,
    NodeStatus,
)
from ..process import ProcessManager, check_health_sync
from ..rpc import RpcWorkerClient
from ..state import get_store

logger = logging.getLogger(__name__)


class LlamaBackend(BaseBackend):
    """
    llama.cpp backend with optional RPC clustering.

    Supports:
    - Local llama-server process management
    - Distributed inference via RPC workers
    - GGUF model format
    - Hot model swapping

    Configuration via environment variables (prefix LLAMA_):
    - LLAMA_PORT: Server port (default: 8080)
    - LLAMA_MODELS_DIR: Models directory (default: ~/.local/share/models)
    - LLAMA_CTX_SIZE: Context size (default: 8192)
    - LLAMA_GPU_LAYERS: GPU layers (default: 99)
    - LLAMA_RPC_WORKERS: Comma-separated list of RPC worker hostnames/IPs
    - LLAMA_RPC_CONTROL_PORT: Control API port on workers (default: 50053)
    - LLAMA_LOAD_TIMEOUT: Model load timeout in seconds (default: 1800)
    - LLAMA_DOWNLOAD_TIMEOUT: Download timeout in seconds (default: 14400)
    """

    def __init__(self, settings: LlamaSettings | None = None):
        self.settings = settings or get_llama_settings()
        super().__init__(
            name="llama",
            display_name="llama.cpp",
            description="Local llama.cpp server with optional RPC clustering. Uses GGUF models.",
            api_base=f"http://localhost:{self.settings.port}",
            health_url=f"http://localhost:{self.settings.port}/health",
            model_format="gguf",
            platforms=["darwin", "linux"],
        )
        self._api_base = f"http://localhost:{self.settings.port}"
        self._health_url = f"http://localhost:{self.settings.port}/health"

        self._process_manager = ProcessManager(graceful_timeout=self.settings.graceful_timeout)
        self._loading = False
        self._error: str | None = None

        # RPC worker client (created lazily if workers are configured)
        self._rpc_client: RpcWorkerClient | None = None
        self._rpc_addresses: str | None = None  # Cached RPC addresses for --rpc flag

    @property
    def api_base(self) -> str:
        return self._api_base

    # ─────────────────────────────────────────────────────────────────
    # Availability
    # ─────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if not super().is_available():
            return False
        # Check if llama-server binary exists
        return shutil.which("llama-server") is not None

    def get_unavailable_reason(self) -> str | None:
        reason = super().get_unavailable_reason()
        if reason:
            return reason
        if shutil.which("llama-server") is None:
            return "llama-server not found in PATH"
        return None

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def start(self) -> tuple[bool, str]:
        """Start the backend (but don't load a model yet).

        llama.cpp doesn't have a separate "start" - it starts when loading a model.
        This just verifies availability.
        """
        logger.info("LlamaBackend: start() called")
        if not self.is_available():
            reason = self.get_unavailable_reason() or "Not available"
            logger.warning(f"LlamaBackend: Not available - {reason}")
            return False, reason
        logger.info("LlamaBackend: Ready (load a model to start inference)")
        return True, "Ready (load a model to start inference)"

    def stop(self) -> tuple[bool, str]:
        """Stop the backend."""
        logger.info("LlamaBackend: stop() called")
        errors = []

        # Stop llama-server
        if self._process_manager.is_running("llama-server"):
            logger.debug("LlamaBackend: Stopping llama-server process")
            success, msg = self._process_manager.stop("llama-server")
            if not success:
                logger.warning(f"LlamaBackend: Failed to stop llama-server: {msg}")
                errors.append(msg)
            else:
                logger.info("LlamaBackend: llama-server stopped")

        # Note: We don't stop RPC workers here - they are managed independently
        # by the user running './toolbox rpc llama' on worker nodes.
        # Workers will be reset before the next model load.

        # Clear state
        get_store().set_loaded_model(self.name, None)
        self._rpc_addresses = None
        self._error = None

        if errors:
            return False, "; ".join(errors)
        logger.info("LlamaBackend: Stopped successfully")
        return True, "Stopped"

    def get_state(self) -> BackendState:
        """Get current backend state."""
        is_running = self._process_manager.is_running("llama-server")
        loaded_model_name = get_store().get_loaded_model(self.name)

        if self._error:
            status = BackendStatus.ERROR
            model_status = ModelStatus.ERROR
        elif self._loading:
            status = BackendStatus.RUNNING
            model_status = ModelStatus.LOADING
        elif is_running and loaded_model_name:
            status = BackendStatus.RUNNING
            model_status = ModelStatus.READY
        elif is_running:
            status = BackendStatus.RUNNING
            model_status = ModelStatus.IDLE
        else:
            status = BackendStatus.STOPPED
            model_status = ModelStatus.IDLE

        loaded_model = None
        if loaded_model_name:
            models = {m.id: m for m in self.list_downloaded_models()}
            loaded_model = models.get(loaded_model_name)

        return BackendState(
            status=status,
            error=self._error,
            model_status=model_status,
            loaded_model=loaded_model,
            model_error=self._error,
            nodes=self.get_cluster_nodes(server_running=is_running),
        )

    def is_healthy(self, timeout: float = 2.0) -> bool:
        """Check if llama-server is responding."""
        healthy = check_health_sync(self._health_url, timeout)
        logger.debug(f"LlamaBackend: Health check -> {healthy}")
        return healthy

    # ─────────────────────────────────────────────────────────────────
    # Model Management
    # ─────────────────────────────────────────────────────────────────

    def list_models(self) -> list[Model]:
        """List all GGUF models in the models directory."""
        models = []
        if not self.settings.models_dir.exists():
            logger.debug(f"LlamaBackend: Models directory does not exist: {self.settings.models_dir}")
            return models

        for path in self.settings.models_dir.glob("*.gguf"):
            # Parse model info from filename
            name = path.stem
            size = path.stat().st_size

            # Try to extract quantization from filename
            quant = ""
            for q in ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16", "F32"]:
                if q.lower() in name.lower():
                    quant = q
                    break

            models.append(Model(
                id=name,
                name=name,
                size_bytes=size,
                format="gguf",
                quantization=quant,
                downloaded=True,
                source=str(path),
            ))

        logger.debug(f"LlamaBackend: Found {len(models)} models")
        return sorted(models, key=lambda m: m.name)

    def list_downloaded_models(self) -> list[Model]:
        """All models are local, so same as list_models."""
        return self.list_models()

    def download_model(self, model_id: str) -> tuple[bool, str]:
        """Download a model.

        model_id can be:
        - A HuggingFace model ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        - A direct URL to a GGUF file
        """
        # TODO: Implement HuggingFace download
        logger.info(f"LlamaBackend: download_model({model_id}) - not implemented")
        return False, (
            f"Automatic download not yet implemented. "
            f"Please download GGUF files manually to {self.settings.models_dir}"
        )

    def delete_model(self, model_id: str) -> tuple[bool, str]:
        """Delete a downloaded model."""
        path = self.settings.models_dir / f"{model_id}.gguf"
        if not path.exists():
            return False, f"Model '{model_id}' not found"

        # Don't delete if it's currently loaded
        if get_store().get_loaded_model(self.name) == model_id:
            return False, f"Cannot delete '{model_id}' while it's loaded"

        try:
            path.unlink()
            logger.info(f"LlamaBackend: Deleted model {model_id}")
            return True, f"Deleted {model_id}"
        except Exception as e:
            logger.error(f"LlamaBackend: Failed to delete {model_id}: {e}")
            return False, f"Failed to delete: {e}"

    def load_model(self, model_id: str) -> tuple[bool, str]:
        """Load a model for inference.

        This is a synchronous, potentially long-running operation.
        It will:
        1. Stop any existing llama-server
        2. Reset RPC workers (if configured) to ensure clean state
        3. Start llama-server with the model (and --rpc flag if workers configured)
        4. Wait for the server to become healthy
        """
        logger.info(f"LlamaBackend: Loading model {model_id}")
        start_time = time.time()

        model_path = self.settings.models_dir / f"{model_id}.gguf"
        if not model_path.exists():
            logger.warning(f"LlamaBackend: Model not found: {model_path}")
            return False, f"Model '{model_id}' not found"

        # Stop existing server if running (managed by us)
        if self._process_manager.is_running("llama-server"):
            logger.debug("LlamaBackend: Stopping existing llama-server")
            self._process_manager.stop("llama-server")
            time.sleep(2)  # Wait for memory release

        # Kill any orphaned llama-server processes (not managed by us)
        logger.debug("LlamaBackend: Killing any orphaned llama-server processes")
        try:
            subprocess.run(["pkill", "-15", "llama-server"], capture_output=True, timeout=5)
            time.sleep(1)
        except Exception:
            pass

        self._loading = True
        self._error = None
        self._rpc_addresses = None

        try:
            # Reset RPC workers if configured
            if self.settings.has_rpc_workers:
                logger.info(f"LlamaBackend: Resetting RPC workers: {self.settings.rpc_worker_list}")
                rpc_result = self._reset_rpc_workers()
                if rpc_result["success"] and rpc_result["rpc_addresses"]:
                    self._rpc_addresses = rpc_result["rpc_addresses"]
                else:
                    # RPC workers unavailable - continue in single-node mode
                    logger.warning(
                        f"LlamaBackend: RPC workers unavailable ({rpc_result['message']}), "
                        "continuing in single-node mode"
                    )

            # Build llama-server command
            cmd = [
                "llama-server",
                "--host", self.settings.host,
                "--port", str(self.settings.port),
                "--model", str(model_path),
                "--n-gpu-layers", self.settings.gpu_layers,
                "--ctx-size", str(self.settings.ctx_size),
            ]

            if self._rpc_addresses:
                cmd.extend(["--rpc", self._rpc_addresses])
                logger.info(f"LlamaBackend: Using RPC workers: {self._rpc_addresses}")

            logger.debug(f"LlamaBackend: Starting llama-server: {' '.join(cmd)}")

            # Start llama-server
            success, msg = self._process_manager.start("llama-server", cmd)
            if not success:
                self._error = msg
                self._loading = False
                logger.error(f"LlamaBackend: Failed to start llama-server: {msg}")
                return False, msg

            # Wait for server to be ready (sync polling)
            logger.info(f"LlamaBackend: Waiting for server to become healthy (timeout: {self.settings.load_timeout}s)")
            if self._wait_for_healthy(timeout=self.settings.load_timeout):
                get_store().set_loaded_model(self.name, model_id)
                self._loading = False
                elapsed = time.time() - start_time
                worker_info = f" (with {len(self.settings.rpc_worker_list)} RPC workers)" if self._rpc_addresses else ""
                logger.info(f"LlamaBackend: Model {model_id} loaded in {elapsed:.1f}s{worker_info}")
                return True, f"Loaded {model_id}{worker_info}"
            else:
                self._process_manager.stop("llama-server", force=True)
                self._error = "Server failed to become healthy"
                self._loading = False
                logger.error(f"LlamaBackend: {self._error}")
                return False, self._error

        except Exception as e:
            self._error = str(e)
            self._loading = False
            logger.exception(f"LlamaBackend: Model load failed: {e}")
            return False, str(e)

    def _reset_rpc_workers(self) -> dict:
        """Reset all RPC workers and get their addresses.

        Returns:
            Dict with keys: success, message, rpc_addresses
        """
        workers = self.settings.rpc_worker_list
        if not workers:
            return {"success": True, "message": "No workers configured", "rpc_addresses": ""}

        logger.info(f"LlamaBackend: Resetting {len(workers)} RPC workers")

        # Create client if needed
        if self._rpc_client is None:
            self._rpc_client = RpcWorkerClient(
                workers=workers,
                control_port=self.settings.rpc_control_port,
                timeout=self.settings.rpc_reset_timeout,
            )

        # Run async reset in sync context
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Reset all workers
                reset_results = loop.run_until_complete(
                    self._rpc_client.reset_all_workers(force=False)
                )

                # Check results
                failed_workers = [host for host, success in reset_results.items() if not success]
                if failed_workers:
                    logger.warning(f"LlamaBackend: Failed to reset workers: {failed_workers}")
                    # Continue with workers that succeeded
                    if all(not success for success in reset_results.values()):
                        return {
                            "success": False,
                            "message": f"All RPC workers failed to reset: {failed_workers}",
                            "rpc_addresses": "",
                        }

                # Get RPC addresses from workers that are running
                rpc_addresses = loop.run_until_complete(
                    self._rpc_client.get_rpc_addresses()
                )

                if not rpc_addresses:
                    return {
                        "success": False,
                        "message": "No RPC workers available after reset",
                        "rpc_addresses": "",
                    }

                successful_count = len(rpc_addresses.split(","))
                logger.info(f"LlamaBackend: {successful_count} RPC workers ready")

                return {
                    "success": True,
                    "message": f"{successful_count} workers ready",
                    "rpc_addresses": rpc_addresses,
                }

            finally:
                loop.close()

        except Exception as e:
            logger.exception(f"LlamaBackend: Failed to reset RPC workers: {e}")
            return {
                "success": False,
                "message": f"Failed to reset RPC workers: {e}",
                "rpc_addresses": "",
            }

    def _wait_for_healthy(self, timeout: float) -> bool:
        """Wait for llama-server to become healthy.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if server became healthy, False on timeout
        """
        deadline = time.time() + timeout
        check_count = 0
        while time.time() < deadline:
            check_count += 1
            if check_health_sync(self._health_url, timeout=2.0):
                logger.debug(f"LlamaBackend: Server healthy after {check_count} checks")
                return True
            if check_count % 10 == 0:
                elapsed = time.time() - (deadline - timeout)
                logger.debug(f"LlamaBackend: Still waiting for healthy... ({elapsed:.0f}s elapsed)")
            time.sleep(0.5)
        logger.warning(f"LlamaBackend: Timeout waiting for healthy after {check_count} checks")
        return False

    def unload_model(self) -> tuple[bool, str]:
        """Unload the current model.

        Note: RPC workers are not stopped - they remain running and will be
        reset before the next model load.
        """
        logger.info("LlamaBackend: Unloading model")
        errors = []

        # Stop llama-server
        if self._process_manager.is_running("llama-server"):
            success, msg = self._process_manager.stop("llama-server")
            if not success:
                logger.warning(f"LlamaBackend: Failed to stop llama-server: {msg}")
                errors.append(msg)
            else:
                logger.info("LlamaBackend: llama-server stopped")
        else:
            logger.debug("LlamaBackend: No llama-server running")

        get_store().set_loaded_model(self.name, None)
        self._rpc_addresses = None

        if errors:
            return False, "; ".join(errors)
        return True, "Model unloaded"

    def get_loaded_model(self) -> Model | None:
        """Get the currently loaded model."""
        model_id = get_store().get_loaded_model(self.name)
        if not model_id:
            return None

        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    # ─────────────────────────────────────────────────────────────────
    # Cluster Management
    # ─────────────────────────────────────────────────────────────────

    def supports_cluster(self) -> bool:
        return True

    def get_cluster_nodes(self, server_running: bool | None = None) -> list[Node]:
        """Get all nodes in the cluster with stats.

        Args:
            server_running: If known, whether the server is running (avoids network check).
                           If None, checks process manager (fast local check).
        """
        nodes = []

        # Use provided state or check process manager (fast, no network)
        if server_running is None:
            server_running = self._process_manager.is_running("llama-server")

        local_status = NodeStatus.ONLINE if server_running else NodeStatus.OFFLINE
        nodes.append(Node(
            id="local",
            hostname="localhost",
            role="main",
            status=local_status,
            **self._collect_local_stats(),
        ))

        # Add RPC worker nodes if configured
        for i, worker_host in enumerate(self.settings.rpc_worker_list):
            worker_status = self._check_worker_status(worker_host)
            nodes.append(Node(
                id=f"worker-{i}",
                hostname=worker_host,
                role="worker",
                status=worker_status,
                # We don't collect detailed stats from workers to keep it fast
                # The control API could be extended to return stats if needed
                gpu_name="",
                gpu_memory_used=0,
                gpu_memory_total=0,
                memory_used=0,
                memory_total=0,
            ))

        return nodes

    def _check_worker_status(self, host: str) -> NodeStatus:
        """Check if a worker's control API is reachable."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, self.settings.rpc_control_port))
            sock.close()
            if result == 0:
                return NodeStatus.ONLINE
            return NodeStatus.OFFLINE
        except Exception:
            return NodeStatus.OFFLINE

    def _collect_local_stats(self) -> dict:
        """Collect memory/GPU stats from local machine."""
        import platform

        stats = {
            "gpu_name": "Metal" if has_metal() else "CPU",
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "memory_used": 0,
            "memory_total": 0,
        }

        try:
            if platform.system() == "Darwin":
                # macOS: total memory
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    stats["memory_total"] = int(result.stdout.strip())
                    # Don't set gpu_memory on macOS - unified memory makes it meaningless

                # Memory usage via vm_stat
                result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    page_size = 16384
                    pages = {"active": 0, "wired": 0, "compressed": 0}
                    for line in result.stdout.split("\n"):
                        if "page size of" in line:
                            page_size = int(line.split()[-2])
                        for key in pages:
                            if f"Pages {key}" in line or f"Pages {key} down" in line:
                                pages[key] = int(line.split(":")[1].strip().rstrip("."))
                    stats["memory_used"] = sum(pages.values()) * page_size

            elif platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            stats["memory_total"] = int(line.split()[1]) * 1024
                        elif line.startswith("MemAvailable:"):
                            stats["memory_used"] = stats["memory_total"] - int(line.split()[1]) * 1024
                            break

                # NVIDIA GPU
                if shutil.which("nvidia-smi"):
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        parts = result.stdout.strip().split(",")
                        if len(parts) >= 3:
                            stats["gpu_name"] = parts[0].strip()
                            stats["gpu_memory_used"] = int(parts[1].strip()) * 1024 * 1024
                            stats["gpu_memory_total"] = int(parts[2].strip()) * 1024 * 1024
        except Exception as e:
            logger.debug(f"LlamaBackend: Failed to collect local stats: {e}")

        return stats

    def configure_cluster(self, config: ClusterConfig) -> tuple[bool, str]:
        """Configure cluster nodes.

        Note: This is a legacy method. The preferred way to configure RPC workers
        is via the LLAMA_RPC_WORKERS environment variable.
        """
        if not config.nodes:
            # Clear worker config
            self.settings.rpc_workers = ""
            self._rpc_client = None
            get_store().set_cluster_config(self.name, {})
            logger.info("LlamaBackend: Cluster configuration cleared")
            return True, "Cluster configuration cleared"

        # Extract hostnames from config
        hostnames = [node.get("hostname", "") for node in config.nodes if node.get("hostname")]
        if not hostnames:
            return False, "No valid hostnames in configuration"

        self.settings.rpc_workers = ",".join(hostnames)
        self._rpc_client = None  # Will be recreated on next use

        get_store().set_cluster_config(self.name, {
            "rpc_workers": self.settings.rpc_workers,
            "rpc_control_port": self.settings.rpc_control_port,
        })

        logger.info(f"LlamaBackend: Configured {len(hostnames)} RPC workers: {hostnames}")
        return True, f"Configured {len(hostnames)} RPC workers"
