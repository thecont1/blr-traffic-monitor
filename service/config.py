"""
Configuration via environment variables.

All settings are read from env vars with sensible defaults for
the docker-compose setup. No .env files needed in the container.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service mode: "api" for HTTP server, "batch" for CLI report generation
    traffic_mode: str = "api"

    # Server binding
    traffic_host: str = "0.0.0.0"
    traffic_port: int = 8000

    # Data directory — contains the CSV files and config/city.json
    data_dir: Path = Path("/data")

    # Output directory — analysis artifacts written here
    output_dir: Path = Path("/output")

    # City config path relative to repo root (not data_dir)
    city_config: str = "config/city.json"

    # Logging
    log_level: str = "info"

    model_config = {"env_prefix": "TRAFFIC_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
