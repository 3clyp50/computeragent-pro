from pydantic import BaseSettings, Field
from typing import Optional

class Settings(BaseSettings):
    MODEL_NAME: str = Field("OS-Copilot/OS-Atlas-Base-7B", env="MODEL_NAME")
    DEVICE_MAP: str = Field("auto", env="DEVICE_MAP")
    TORCH_DTYPE: str = Field("auto", env="TORCH_DTYPE")
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")

    # Domain and environment settings
    DOMAIN_NAME: str = Field("computeragent.pro", env="DOMAIN_NAME")
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")  # or "production"

    # Security
    AI_AGENT_KEY: Optional[str] = Field(None, env="AI_AGENT_KEY")  # Optional: Set in production

    # Logging
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"  # Specify a .env filename here if you want

# Instantiate settings
settings = Settings()