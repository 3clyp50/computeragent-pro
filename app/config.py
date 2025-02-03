from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os

class Settings(BaseSettings):
    MODEL_NAME: str = Field("OS-Copilot/OS-Atlas-Base-7B", env="MODEL_NAME")
    DEVICE_MAP: str = Field("auto", env="DEVICE_MAP")
    TORCH_DTYPE: str = Field("auto", env="TORCH_DTYPE")
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")

    # Domain and environment settings
    DOMAIN_NAME: str = Field("api.computeragent.pro", env="DOMAIN_NAME")
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")  # or "production"

    # Security
    AI_AGENT_KEYS: Optional[str] = Field(None, env="AI_AGENT_KEYS")  # Comma-separated API keys

    # Model cache
    HF_HOME: str = Field("/app/model_cache", env="HF_HOME", description="Must be absolute path in container")

    # Logging
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"  # Specify a .env filename here if you want

    @property
    def ai_agent_keys_list(self) -> List[str]:
        """Parse AI_AGENT_KEYS into a list."""
        if self.AI_AGENT_KEYS:
            return [key.strip() for key in self.AI_AGENT_KEYS.split(",") if key.strip()]
        return []

# Instantiate settings
settings = Settings()
