from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml


class Settings(BaseSettings):
    tiingo_api_key: str = ""
    news_api_key: str = ""
    hf_token: str = ""
    ntfy_base_url: str = "https://ntfy.sh"
    ntfy_topic: str = "trading-copilot"
    webhook_base_url: str = "http://localhost:8000"
    duckdb_path: str = "data/market.duckdb"
    sqlite_path: str = "data/positions.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


settings = Settings()
config = load_config() if Path("config.yaml").exists() else {}
