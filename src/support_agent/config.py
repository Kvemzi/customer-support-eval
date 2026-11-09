"""
Centralised, typed configuration.

Every environment variable is read in exactly ONE place. Other modules
receive a `Settings` object (dependency injection) instead of calling
os.getenv() themselves, which makes them easy to test with custom settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# src/support_agent/config.py -> parents[2] == project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SOURCE_KB_PATH: Path = PROJECT_ROOT / "data" / "knowledge_base" / "policies.json"
DEFAULT_KB_PATH: Path = (
    SOURCE_KB_PATH if SOURCE_KB_PATH.is_file()
    else Path(str(files("shopco_policy_data").joinpath("policies.json")))
)

# Load .env from the project root regardless of the current working directory.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration of the SUT."""

    backend: str = "auto"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    memory_max_turns: int = 10
    retrieval_top_k: int = 2
    kb_path: Path = DEFAULT_KB_PATH
    http_endpoint: str = "http://localhost:8000/chat"
    http_timeout: int = 30
    openai_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables (with safe defaults)."""
        kb_path_env = os.getenv("SUT_KB_PATH")
        kb_path = Path(kb_path_env) if kb_path_env else DEFAULT_KB_PATH
        if not kb_path.is_absolute():
            kb_path = PROJECT_ROOT / kb_path

        return cls(
            backend=os.getenv("SUT_BACKEND", "auto").lower(),
            model=os.getenv("SUT_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("SUT_TEMPERATURE", "0.3")),
            memory_max_turns=int(os.getenv("SUT_MEMORY_MAX_TURNS", "10")),
            retrieval_top_k=int(os.getenv("SUT_RETRIEVAL_TOP_K", "2")),
            kb_path=kb_path,
            http_endpoint=os.getenv("SUT_HTTP_ENDPOINT", "http://localhost:8000/chat"),
            http_timeout=int(os.getenv("SUT_HTTP_TIMEOUT", "30")),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        )