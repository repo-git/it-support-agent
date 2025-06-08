# src/utils/config.py
"""
Configuration manager per IT Support Agent
"""
import os
from typing import Optional
from pydantic import BaseModel
from enum import Enum


class AIProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"


class Config(BaseModel):
    """Configurazione principale dell'applicazione"""

    # LiveKit Configuration
    livekit_url: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "devkey")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "")

    # AI Configuration
    default_ai_provider: AIProvider = AIProvider(os.getenv("DEFAULT_AI_PROVIDER", "ollama"))

    # Ollama Configuration
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3:latest")

    # OpenAI Configuration
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4-vision-preview")

    # Gemini Configuration
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-pro-vision")

    # Claude Configuration
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")

    # Redis Configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Application Settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    max_session_duration: int = int(os.getenv("MAX_SESSION_DURATION", "3600"))
    screenshot_interval: int = int(os.getenv("SCREENSHOT_INTERVAL", "30"))

    # Security Settings
    session_secret: str = os.getenv("SESSION_SECRET", "default-secret")
    allowed_hosts: list = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

    # Remote Control Settings
    enable_remote_control: bool = os.getenv("ENABLE_REMOTE_CONTROL", "true").lower() == "true"
    require_approval: bool = os.getenv("REQUIRE_APPROVAL", "true").lower() == "true"
    control_timeout: int = int(os.getenv("CONTROL_TIMEOUT", "300"))
