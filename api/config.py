from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    ollama_host: str = "http://localhost:11434"
    model_dir: Path = Path("./models")
    log_level: str = "INFO"
    max_video_size_mb: int = 500
    default_person_height_cm: float = 170.0
    reports_output_dir: Path = Path("./output/reports")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
