import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATABASE_URL = "postgresql+psycopg://cfb:cfb@localhost:5432/cfb"
DEFAULT_CACHE_DIR = Path("data/raw")


@dataclass(frozen=True)
class Settings:
    cfbd_api_key: str | None
    database_url: str
    cache_dir: Path


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        cfbd_api_key=os.environ.get("CFBD_API_KEY") or None,
        database_url=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        cache_dir=Path(os.environ.get("CACHE_DIR", DEFAULT_CACHE_DIR)),
    )
