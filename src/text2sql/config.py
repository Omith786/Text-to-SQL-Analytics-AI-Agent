"""Runtime configuration, driven entirely by environment variables.

The agent talks to any OpenAI-compatible chat-completions endpoint. The default
target is a local Ollama server, which is free and needs no account. Hosted
free-tier providers (for example Groq or Google's OpenAI-compatible gateway) can
be selected purely by pointing the base URL, key and model at them; no provider
is hard-coded beyond these defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repository-relative default location of the bundled sample database.
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sample.duckdb"

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_dotenv(path: Path = _ENV_PATH) -> None:
    """Populate ``os.environ`` from a ``.env`` file, without overriding existing
    values. A minimal ``KEY=value`` parser so no extra dependency is needed;
    lines that are blank or start with ``#`` are ignored.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class LLMConfig:
    """Connection settings for an OpenAI-compatible chat endpoint."""

    base_url: str
    api_key: str
    model: str
    temperature: float
    request_timeout: float

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_dotenv()
        return cls(
            # Ollama exposes an OpenAI-compatible API at /v1 on port 11434.
            base_url=os.getenv("T2SQL_BASE_URL", "http://localhost:11434/v1"),
            # Ollama ignores the key, but the OpenAI client requires a non-empty value.
            api_key=os.getenv("T2SQL_API_KEY", "ollama"),
            model=os.getenv("T2SQL_MODEL", "qwen2.5:1.5b"),
            temperature=float(os.getenv("T2SQL_TEMPERATURE", "0.0")),
            request_timeout=float(os.getenv("T2SQL_TIMEOUT", "120")),
        )


@dataclass(frozen=True)
class AgentConfig:
    """Behavioural limits for the tool-using loop and the SQL sandbox."""

    max_retries: int = int(os.getenv("T2SQL_MAX_RETRIES", "3"))
    row_limit: int = int(os.getenv("T2SQL_ROW_LIMIT", "1000"))
    statement_timeout_s: float = float(os.getenv("T2SQL_SQL_TIMEOUT", "10"))
