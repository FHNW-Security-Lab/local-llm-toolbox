"""llama.cpp backend implementation."""

import asyncio
import logging
import shutil

import asyncssh

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
from ..process import ProcessManager, wait_for_healthy, check_health_sync
from ..state import get_store

logger = logging.getLogger(__name__)


class LlamaBackend(BaseBackend):
    """
    llama.cpp backend with optional RPC clustering.

    Supports:
    - Local llama-server process management
    - Remote RPC servers via SSH
    - GGUF model format
    - Hot model swapping

    Configuration via environment variables (prefix LLAMA_):
    - LLAMA_PORT: Server port (default: 8080)
    - LLAMA_MODELS_DIR: Models directory (default: ~/.local/share/models)
    - LLAMA_CTX_SIZE: Context size (default: 8192)
    - LLAMA_GPU_LAYERS: GPU layers (default: 99)
    - LLAMA_RPC_HOST: Remote RPC hostname
    - LLAMA_SSH_USER: SSH username for remote nodes
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
        self._ssh_conn: asyncssh.SSHClientConnection | None = None
        self._loading = False
        self._error: str | None = None

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
        """Start the backend (but don't load a model yet)."""
        # llama.cpp doesn't have a separate "start" - it starts when loading a model
        # For now, just verify we can run
        if not self.is_available():
            return False, self.get_unavailable_reason() or "Not available"
        return True, "Ready (load a model to start inference)"

    def stop(self) -> tuple[bool, str]:
        """Stop the backend."""
        return asyncio.run(self._stop_async())

    async def _stop_async(self) -> tuple[bool, str]:
        """Async stop implementation."""
        errors = []

        # Stop llama-server
        if self._process_manager.is_running("llama-server"):
            success, msg = self._process_manager.stop("llama-server")
            if not success:
                errors.append(msg)

        # Stop remote RPC server
        if self.settings.has_remote:
            if not await self._stop_rpc_server():
                errors.append("Failed to stop RPC server")

        # Close SSH connection
        await self._close_ssh()

        # Clear state
        get_store().set_loaded_model(self.name, None)
        self._error = None

        if errors:
            return False, "; ".join(errors)
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
            nodes=self.get_cluster_nodes(),
        )

    def is_healthy(self, timeout: float = 2.0) -> bool:
        """Check if llama-server is responding."""
        return check_health_sync(self._health_url, timeout)

    # ─────────────────────────────────────────────────────────────────
    # Model Management
    # ─────────────────────────────────────────────────────────────────

    def list_models(self) -> list[Model]:
        """List all GGUF models in the models directory."""
        models = []
        if not self.settings.models_dir.exists():
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

        return sorted(models, key=lambda m: m.name)

    def list_downloaded_models(self) -> list[Model]:
        """All models are local, so same as list_models."""
        return self.list_models()

    def download_model(self, model_id: str) -> tuple[bool, str]:
        """
        Download a model.

        model_id can be:
        - A HuggingFace model ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        - A direct URL to a GGUF file
        """
        # TODO: Implement HuggingFace download
        # For now, just provide instructions
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
            return True, f"Deleted {model_id}"
        except Exception as e:
            return False, f"Failed to delete: {e}"

    def load_model(self, model_id: str) -> tuple[bool, str]:
        """Load a model for inference."""
        return asyncio.run(self._load_model_async(model_id))

    async def _load_model_async(self, model_id: str) -> tuple[bool, str]:
        """Async model loading."""
        import subprocess

        model_path = self.settings.models_dir / f"{model_id}.gguf"
        if not model_path.exists():
            return False, f"Model '{model_id}' not found"

        # Stop existing server if running (managed by us)
        if self._process_manager.is_running("llama-server"):
            self._process_manager.stop("llama-server")
            await asyncio.sleep(2)  # Wait for memory release

        # Kill any orphaned llama-server processes (not managed by us)
        try:
            subprocess.run(["pkill", "-15", "llama-server"], capture_output=True, timeout=5)
            await asyncio.sleep(1)
        except Exception:
            pass

        self._loading = True
        self._error = None

        try:
            # Start RPC server if configured
            if self.settings.has_remote:
                if not await self._start_rpc_server():
                    self._error = "Failed to start RPC server"
                    return False, self._error

            # Build llama-server command
            cmd = [
                "llama-server",
                "--host", self.settings.host,
                "--port", str(self.settings.port),
                "--model", str(model_path),
                "--n-gpu-layers", str(self.settings.gpu_layers),
                "--ctx-size", str(self.settings.ctx_size),
            ]

            if self.settings.has_remote:
                cmd.extend(["--rpc", self.settings.rpc_address])

            # Start llama-server
            success, msg = self._process_manager.start("llama-server", cmd)
            if not success:
                self._error = msg
                return False, msg

            # Wait for server to be ready
            if await wait_for_healthy(self._health_url, timeout=self.settings.load_timeout):
                get_store().set_loaded_model(self.name, model_id)
                self._loading = False
                return True, f"Loaded {model_id}"
            else:
                self._process_manager.stop("llama-server", force=True)
                self._error = "Server failed to become healthy"
                self._loading = False
                return False, self._error

        except Exception as e:
            self._error = str(e)
            self._loading = False
            logger.exception("Model load failed")
            return False, str(e)

    def unload_model(self) -> tuple[bool, str]:
        """Unload the current model."""
        if not self._process_manager.is_running("llama-server"):
            get_store().set_loaded_model(self.name, None)
            return True, "No model loaded"

        success, msg = self._process_manager.stop("llama-server")
        if success:
            get_store().set_loaded_model(self.name, None)
        return success, msg

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

    def get_cluster_nodes(self) -> list[Node]:
        """Get all nodes in the cluster with stats."""
        nodes = []

        # Local node with stats
        local_status = NodeStatus.ONLINE if self.is_healthy() else NodeStatus.OFFLINE
        nodes.append(Node(
            id="local",
            hostname="localhost",
            role="local",
            status=local_status,
            **self._collect_local_stats(),
        ))

        # Remote node if configured
        if self.settings.has_remote:
            remote_status, remote_stats = self._collect_remote_stats()
            nodes.append(Node(
                id="remote",
                hostname=self.settings.rpc_host,
                role="remote",
                status=remote_status,
                **remote_stats,
            ))

        return nodes

    def _collect_local_stats(self) -> dict:
        """Collect memory/GPU stats from local machine."""
        import platform
        import subprocess

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
            logger.debug(f"Failed to collect local stats: {e}")

        return stats

    def _collect_remote_stats(self) -> tuple[NodeStatus, dict]:
        """Collect stats from remote node via SSH."""
        import socket
        import subprocess

        stats = {
            "gpu_name": "",
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "memory_used": 0,
            "memory_total": 0,
        }

        # Check RPC port reachability
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            if sock.connect_ex((self.settings.rpc_host, self.settings.rpc_port)) != 0:
                sock.close()
                return NodeStatus.OFFLINE, stats
            sock.close()
        except Exception:
            return NodeStatus.OFFLINE, stats

        # Get stats via SSH
        try:
            cmd = (
                "cat /proc/meminfo 2>/dev/null | grep -E '^(MemTotal|MemAvailable):'; "
                "nvidia-smi --query-gpu=name,memory.used,memory.total "
                "--format=csv,noheader,nounits 2>/dev/null || true"
            )
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                 "-p", str(self.settings.ssh_port),
                 f"{self.settings.ssh_user}@{self.settings.rpc_host}", cmd],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("MemTotal:"):
                        stats["memory_total"] = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        stats["memory_used"] = stats["memory_total"] - int(line.split()[1]) * 1024
                    elif "," in line and not line.startswith("Mem"):
                        parts = line.split(",")
                        if len(parts) >= 3:
                            stats["gpu_name"] = parts[0].strip()
                            stats["gpu_memory_used"] = int(parts[1].strip()) * 1024 * 1024
                            stats["gpu_memory_total"] = int(parts[2].strip()) * 1024 * 1024
        except Exception as e:
            logger.debug(f"Failed to collect remote stats: {e}")

        return NodeStatus.ONLINE, stats

    def configure_cluster(self, config: ClusterConfig) -> tuple[bool, str]:
        """Configure cluster nodes."""
        if not config.nodes:
            # Clear remote config
            self.settings.rpc_host = ""
            self.settings.ssh_user = ""
            get_store().set_cluster_config(self.name, {})
            return True, "Cluster configuration cleared"

        # For now, support single remote node
        if len(config.nodes) > 1:
            return False, "Only single remote node supported currently"

        node = config.nodes[0]
        self.settings.rpc_host = node.get("hostname", "")
        self.settings.ssh_user = node.get("user", "")
        self.settings.rpc_port = node.get("rpc_port", 50052)
        self.settings.ssh_port = node.get("ssh_port", 22)

        get_store().set_cluster_config(self.name, {
            "rpc_host": self.settings.rpc_host,
            "ssh_user": self.settings.ssh_user,
            "rpc_port": self.settings.rpc_port,
            "ssh_port": self.settings.ssh_port,
        })

        return True, f"Configured remote node: {self.settings.rpc_host}"

    # ─────────────────────────────────────────────────────────────────
    # SSH / RPC Management
    # ─────────────────────────────────────────────────────────────────

    async def _get_ssh(self) -> asyncssh.SSHClientConnection:
        """Get or create SSH connection."""
        if self._ssh_conn is not None:
            try:
                await asyncio.wait_for(self._ssh_conn.run("true"), timeout=5)
                return self._ssh_conn
            except Exception:
                await self._close_ssh()

        logger.info(f"SSH connecting to {self.settings.rpc_host}...")
        self._ssh_conn = await asyncio.wait_for(
            asyncssh.connect(
                self.settings.rpc_host,
                port=self.settings.ssh_port,
                username=self.settings.ssh_user,
                known_hosts=None,
            ),
            timeout=self.settings.ssh_timeout,
        )
        return self._ssh_conn

    async def _close_ssh(self):
        """Close SSH connection."""
        if self._ssh_conn:
            self._ssh_conn.close()
            await self._ssh_conn.wait_closed()
            self._ssh_conn = None

    async def _run_remote(self, cmd: str, timeout: int = 30) -> bool:
        """Run command on remote node."""
        if not self.settings.has_remote:
            return True
        try:
            ssh = await self._get_ssh()
            result = await asyncio.wait_for(ssh.run(cmd), timeout=timeout)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"SSH command failed: {e}")
            return False

    async def _start_rpc_server(self) -> bool:
        """Start RPC server on remote node."""
        if not self.settings.has_remote:
            return True

        logger.info("Starting remote rpc-server...")
        await self._run_remote("pkill -15 rpc-server || true")
        await asyncio.sleep(1)

        cmd = f"nohup rpc-server --host 0.0.0.0 --port {self.settings.rpc_port} > /tmp/rpc-server.log 2>&1 &"
        await self._run_remote(cmd)

        # Wait for RPC port
        for _ in range(60):
            try:
                r, w = await asyncio.wait_for(
                    asyncio.open_connection(self.settings.rpc_host, self.settings.rpc_port),
                    timeout=2,
                )
                w.close()
                await w.wait_closed()
                logger.info("RPC server ready")
                return True
            except (OSError, asyncio.TimeoutError):
                await asyncio.sleep(1)

        logger.error("RPC server failed to start")
        return False

    async def _stop_rpc_server(self, force: bool = False) -> bool:
        """Stop RPC server on remote node."""
        if not self.settings.has_remote:
            return True
        sig = "-9" if force else "-15"
        logger.info(f"Stopping rpc-server (force={force})")
        return await self._run_remote(f"pkill {sig} rpc-server || true")
