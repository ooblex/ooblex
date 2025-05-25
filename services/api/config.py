"""
Configuration management using Pydantic settings
"""
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # General
    app_name: str = "Ooblex API"
    debug: bool = False
    log_level: str = "INFO"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8800
    api_workers: int = 4
    
    # Security
    jwt_secret: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    
    # Database
    database_url: str = "postgresql://ooblex:ooblex@localhost:5432/ooblex"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # RabbitMQ
    rabbitmq_url: str = "amqp://admin:admin@localhost:5672"
    
    # CORS
    cors_origins: str = "*"
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    # Features
    enable_face_swap: bool = True
    enable_object_detection: bool = True
    enable_style_transfer: bool = True
    enable_background_removal: bool = True
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    telemetry_enabled: bool = False
    
    # External Services
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_s3_bucket: Optional[str] = None
    aws_region: str = "us-east-1"


# Create settings instance
settings = Settings()