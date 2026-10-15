"""Central configuration loaded from .env."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    ollama_url: str
    ollama_model: str
    embed_model: str
    rerank_model: str
    rerank: bool
    chunk_size: int
    chunk_overlap: int
    top_k: int


settings = Settings(
    gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
    gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
    ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
    ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip(),
    embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip(),
    rerank_model=os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2").strip(),
    rerank=os.getenv("RERANK", "1").strip() not in ("0", "false", "no", ""),
    chunk_size=_env_int("CHUNK_SIZE", 1200),
    chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
    top_k=_env_int("TOP_K", 8),
)

DATA_DIR = Path(os.getenv("DATA_DIR") or BASE_DIR / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "library.db"
PDF_DIR = DATA_DIR / "pdfs"  # original PDFs, kept so cited pages can be shown
PDF_DIR.mkdir(exist_ok=True)
