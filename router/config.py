"""Router configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROUTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 5001  # 5000 is often used by AirPlay on macOS
    health_interval: float = 5.0


config = Config()
