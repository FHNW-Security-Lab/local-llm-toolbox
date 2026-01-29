"""Configuration management with environment variable overrides."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlamaSettings(BaseSettings):
    """
    Configuration for llama.cpp backend.

    All settings can be overridden via environment variables with LLAMA_ prefix.
    Example: LLAMA_PORT=8081 to change the default port.
    """
    model_config = SettingsConfigDict(
        env_prefix="LLAMA_",
        env_file=".env",
        extra="ignore",
    )

    # Model storage
    models_dir: Path = Field(
        default_factory=lambda: Path.home() / ".local/share/models",
        description="Directory where GGUF models are stored",
    )

    # llama-server settings
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8080, description="Port for the llama-server")
    ctx_size: int = Field(default=8192, description="Context size in tokens")
    gpu_layers: int = Field(default=99, description="Number of layers to offload to GPU")

    # RPC cluster settings
    rpc_host: str = Field(default="", description="Remote RPC server hostname")
    rpc_port: int = Field(default=50052, description="RPC server port")
    ssh_user: str = Field(default="", description="SSH username for remote nodes")
    ssh_port: int = Field(default=22, description="SSH port for remote nodes")

    # Timeouts (in seconds)
    # Large models (70B+) can take 2-3+ hours to download on slower connections
    download_timeout: int = Field(
        default=14400,  # 4 hours
        description="Timeout for model downloads (seconds)",
    )
    load_timeout: int = Field(
        default=1800,  # 30 minutes
        description="Timeout for model loading (seconds)",
    )
    graceful_timeout: int = Field(
        default=15,
        description="Graceful shutdown timeout (seconds)",
    )
    ssh_timeout: int = Field(
        default=30,
        description="SSH connection timeout (seconds)",
    )
    health_check_interval: int = Field(
        default=10,
        description="Interval between health checks (seconds)",
    )

    @property
    def has_remote(self) -> bool:
        """Check if remote RPC is configured."""
        return bool(self.rpc_host and self.ssh_user)

    @property
    def rpc_address(self) -> str:
        """Get the full RPC address."""
        if not self.rpc_host:
            return ""
        return f"{self.rpc_host}:{self.rpc_port}"


class FoundrySettings(BaseSettings):
    """
    Configuration for Microsoft Foundry Local backend.

    All settings can be overridden via environment variables with FOUNDRY_ prefix.
    """
    model_config = SettingsConfigDict(
        env_prefix="FOUNDRY_",
        env_file=".env",
        extra="ignore",
    )

    port: int = Field(default=5273, description="Port for the Foundry server")

    # Timeouts
    download_timeout: int = Field(
        default=14400,  # 4 hours
        description="Timeout for model downloads (seconds)",
    )
    load_timeout: int = Field(
        default=1800,
        description="Timeout for model loading (seconds)",
    )
    graceful_timeout: int = Field(
        default=15,
        description="Graceful shutdown timeout (seconds)",
    )


class VllmSettings(BaseSettings):
    """
    Configuration for vLLM backend.

    All settings can be overridden via environment variables with VLLM_ prefix.
    """
    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port for the vLLM server")

    # vLLM specific
    tensor_parallel_size: int = Field(default=1, description="Number of GPUs for tensor parallelism")
    gpu_memory_utilization: float = Field(default=0.9, description="GPU memory utilization fraction")

    # Timeouts
    download_timeout: int = Field(
        default=14400,
        description="Timeout for model downloads (seconds)",
    )
    load_timeout: int = Field(
        default=1800,
        description="Timeout for model loading (seconds)",
    )


class SglangSettings(BaseSettings):
    """
    Configuration for SGLang backend.

    All settings can be overridden via environment variables with SGLANG_ prefix.
    """
    model_config = SettingsConfigDict(
        env_prefix="SGLANG_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=30000, description="Port for the SGLang server")

    # Timeouts
    download_timeout: int = Field(
        default=14400,
        description="Timeout for model downloads (seconds)",
    )
    load_timeout: int = Field(
        default=1800,
        description="Timeout for model loading (seconds)",
    )


class RouterSettings(BaseSettings):
    """
    Configuration for the API router.

    All settings can be overridden via environment variables with ROUTER_ prefix.
    """
    model_config = SettingsConfigDict(
        env_prefix="ROUTER_",
        env_file=".env",
        extra="ignore",
    )

    port: int = Field(default=5001, description="Port for the router")  # 5000 used by AirPlay
    health_interval: int = Field(default=5, description="Backend health check interval (seconds)")


class DashboardSettings(BaseSettings):
    """
    Configuration for the web dashboard.

    All settings can be overridden via environment variables with DASHBOARD_ prefix.
    """
    model_config = SettingsConfigDict(
        env_prefix="DASHBOARD_",
        env_file=".env",
        extra="ignore",
    )

    port: int = Field(default=8090, description="Port for the dashboard")


# Singleton instances (created on first access)
_llama_settings: LlamaSettings | None = None
_foundry_settings: FoundrySettings | None = None
_vllm_settings: VllmSettings | None = None
_sglang_settings: SglangSettings | None = None
_router_settings: RouterSettings | None = None
_dashboard_settings: DashboardSettings | None = None


def get_llama_settings() -> LlamaSettings:
    """Get llama.cpp backend settings."""
    global _llama_settings
    if _llama_settings is None:
        _llama_settings = LlamaSettings()
    return _llama_settings


def get_foundry_settings() -> FoundrySettings:
    """Get Foundry Local backend settings."""
    global _foundry_settings
    if _foundry_settings is None:
        _foundry_settings = FoundrySettings()
    return _foundry_settings


def get_vllm_settings() -> VllmSettings:
    """Get vLLM backend settings."""
    global _vllm_settings
    if _vllm_settings is None:
        _vllm_settings = VllmSettings()
    return _vllm_settings


def get_sglang_settings() -> SglangSettings:
    """Get SGLang backend settings."""
    global _sglang_settings
    if _sglang_settings is None:
        _sglang_settings = SglangSettings()
    return _sglang_settings


def get_router_settings() -> RouterSettings:
    """Get router settings."""
    global _router_settings
    if _router_settings is None:
        _router_settings = RouterSettings()
    return _router_settings


def get_dashboard_settings() -> DashboardSettings:
    """Get dashboard settings."""
    global _dashboard_settings
    if _dashboard_settings is None:
        _dashboard_settings = DashboardSettings()
    return _dashboard_settings
