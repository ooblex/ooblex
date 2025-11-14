"""
ML Worker configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """ML Worker settings"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Redis
    redis_url: str = "redis://localhost:6379"

    # RabbitMQ
    rabbitmq_url: str = "amqp://admin:admin@localhost:5672"

    # Model Configuration
    model_path: str = "/models"
    model_cache_size: int = 5

    # GPU Configuration
    cuda_visible_devices: str = "0"
    tf_cpp_min_log_level: str = "2"

    # Logging
    log_level: str = "INFO"


settings = Settings()
