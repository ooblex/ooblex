"""
WebRTC Gateway configuration
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """WebRTC Gateway settings"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # WebRTC
    webrtc_port: int = 8100
    webrtc_stun_servers: str = "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302"
    webrtc_turn_server: str = ""
    webrtc_turn_username: str = ""
    webrtc_turn_password: str = ""
    
    # SSL
    ssl_cert_path: str = "/ssl/cert.pem"
    ssl_key_path: str = "/ssl/key.pem"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # RabbitMQ
    rabbitmq_url: str = "amqp://admin:admin@localhost:5672"
    
    # Logging
    log_level: str = "INFO"
    
    # Media
    max_video_bitrate: int = 2000000  # 2 Mbps
    max_audio_bitrate: int = 128000   # 128 Kbps
    video_codec: str = "VP8"
    audio_codec: str = "OPUS"


settings = Settings()