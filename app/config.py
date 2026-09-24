from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Small dependency-free dotenv loader; existing environment wins."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_chat_path: str = os.getenv("LLM_CHAT_PATH", "")
    llm_model: str = os.getenv("LLM_MODEL", "auto")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_api_key_header: str = os.getenv("LLM_API_KEY_HEADER", "Authorization")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
    llm_concurrency: int = int(os.getenv("LLM_CONCURRENCY", "3"))
    # Set this to the model's confirmed total context window. Zero disables
    # local preflight validation when the platform does not expose that value.
    llm_context_window_tokens: int = int(os.getenv("LLM_CONTEXT_WINDOW_TOKENS", "0"))
    llm_context_safety_margin_tokens: int = int(os.getenv("LLM_CONTEXT_SAFETY_MARGIN_TOKENS", "512"))
    # Raw model output can contain transaction-derived personal data. Keep this
    # off by default and enable only in an approved, access-controlled log sink.
    llm_log_raw_response: bool = os.getenv("LLM_LOG_RAW_RESPONSE", "false").lower() == "true"
    llm_log_raw_response_max_chars: int = int(os.getenv("LLM_LOG_RAW_RESPONSE_MAX_CHARS", "20000"))
    llm_queue_maxsize: int = int(os.getenv("LLM_QUEUE_MAXSIZE", "60"))
    llm_queue_enqueue_timeout_seconds: float = float(os.getenv("LLM_QUEUE_ENQUEUE_TIMEOUT_SECONDS", "5"))
    customer_info_parquet_path: str = os.getenv(
        "CUSTOMER_INFO_PARQUET_PATH",
        str(BASE_DIR / "data" / "customer_info.parquet"), 
    )
    max_response_tokens: int = int(os.getenv("MAX_RESPONSE_TOKENS", "2200"))
    # Zero omits the parameter; only enable if the loader accepts frequency_penalty.
    llm_frequency_penalty: float = float(os.getenv("LLM_FREQUENCY_PENALTY", "0"))
    report_directory: str = os.getenv("REPORT_DIRECTORY", str(BASE_DIR / "data" / "reports"))
    require_human_review: bool = os.getenv("REQUIRE_HUMAN_REVIEW", "true").lower() == "true"

    def __post_init__(self) -> None:
        positive_values = {
            "LLM_TIMEOUT_SECONDS": self.llm_timeout_seconds,
            "LLM_CONCURRENCY": self.llm_concurrency,
            "LLM_QUEUE_MAXSIZE": self.llm_queue_maxsize,
            "LLM_QUEUE_ENQUEUE_TIMEOUT_SECONDS": self.llm_queue_enqueue_timeout_seconds,
            "MAX_RESPONSE_TOKENS": self.max_response_tokens,
            "LLM_LOG_RAW_RESPONSE_MAX_CHARS": self.llm_log_raw_response_max_chars,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be greater than zero: {', '.join(invalid)}")
        if self.llm_context_window_tokens < 0:
            raise ValueError("LLM_CONTEXT_WINDOW_TOKENS must be zero or greater")
        if self.llm_context_safety_margin_tokens < 0:
            raise ValueError("LLM_CONTEXT_SAFETY_MARGIN_TOKENS must be zero or greater")
        if self.llm_context_window_tokens and self.llm_prompt_token_budget <= 0:
            raise ValueError(
                "LLM_CONTEXT_WINDOW_TOKENS must exceed MAX_RESPONSE_TOKENS plus "
                "LLM_CONTEXT_SAFETY_MARGIN_TOKENS"
            )

    @property
    def chat_url(self) -> str:
        return f"{self.llm_base_url.rstrip('/')}/{self.llm_chat_path.lstrip('/')}"

    @property
    def report_directory_path(self) -> Path:
        path = Path(self.report_directory)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def customer_info_path(self) -> Path:
        path = Path(self.customer_info_parquet_path)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def llm_prompt_token_budget(self) -> int | None:
        if not self.llm_context_window_tokens:
            return None
        return (
            self.llm_context_window_tokens
            - self.max_response_tokens
            - self.llm_context_safety_margin_tokens
        )