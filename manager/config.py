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

    # RPC cluster settings (for distributed inference)
    # Workers are specified as comma-separated hostnames/IPs
    # Example: LLAMA_RPC_WORKERS=192.168.1.10,192.168.1.11
    rpc_workers: str = Field(
        default="",
        description="Comma-separated list of RPC worker hostnames/IPs",
    )
    rpc_control_port: int = Field(
        default=50053,
        description="Control API port on workers (same for all workers)",
    )

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
    rpc_reset_timeout: int = Field(
        default=30,
        description="Timeout for resetting RPC workers (seconds)",
    )
    health_check_interval: int = Field(
        default=10,
        description="Interval between health checks (seconds)",
    )

    @property
    def has_rpc_workers(self) -> bool:
        """Check if RPC workers are configured."""
        return bool(self.rpc_workers.strip())

    @property
    def rpc_worker_list(self) -> list[str]:
        """Get list of RPC worker hostnames/IPs."""
        if not self.rpc_workers.strip():
            return []
        return [w.strip() for w in self.rpc_workers.split(",") if w.strip()]


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


# Singleton instances (created on first access)
_llama_settings: LlamaSettings | None = None
_foundry_settings: FoundrySettings | None = None
_vllm_settings: VllmSettings | None = None
_sglang_settings: SglangSettings | None = None


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
