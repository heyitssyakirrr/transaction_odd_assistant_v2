from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent # change based on where u put the file


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
    # CSV with columns CODE, D_CUST_OCCUPAT -- maintained by compliance,
    # loaded and cached at analysis time (see app/core/reference_data.py).
    occupation_code_csv_path: str = os.getenv(
        "OCCUPATION_CODE_CSV_PATH",
        str(BASE_DIR / "data" / "occupation_codes.csv"),
    )
    # The compact single-account decision has a deliberately small completion
    # budget. A larger limit lets a misconfigured chat template run away.
    max_response_tokens: int = int(os.getenv("MAX_RESPONSE_TOKENS", "1200"))
    # Qwen's low-variance sampling defaults. Set only parameters accepted by
    # the bank's OpenAI-compatible loader.
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    llm_top_p: float = float(os.getenv("LLM_TOP_P", "0.001"))
    # This is a vLLM/Qwen extension rather than a portable OpenAI parameter.
    # Leave disabled unless the bank loader is confirmed to accept it.
    llm_repetition_penalty: float = float(os.getenv("LLM_REPETITION_PENALTY", "0"))
    # "json_schema" is the current OpenAI/vLLM contract. Older vLLM loaders
    # may require "guided_json" instead; only use it after platform confirms
    # that request field is supported.
    llm_structured_output_protocol: str = os.getenv("LLM_STRUCTURED_OUTPUT_PROTOCOL", "json_schema").lower()
    # Qwen's native end token and the literal role labels seen when a loader
    # has an incomplete chat template. These stop generation after a complete
    # answer; they are never accepted as part of the JSON response.
    llm_stop_sequences: tuple[str, ...] = tuple(
        value for value in os.getenv(
            "LLM_STOP_SEQUENCES", "<|im_end|>|<|endoftext|>|system:|user:"
        ).split("|") if value
    )
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
        if not 0 <= self.llm_temperature <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if not 0 < self.llm_top_p <= 1:
            raise ValueError("LLM_TOP_P must be greater than 0 and no greater than 1")
        if self.llm_repetition_penalty < 0:
            raise ValueError("LLM_REPETITION_PENALTY must not be negative")
        if self.llm_structured_output_protocol not in {"json_schema", "guided_json", "json_object", "off"}:
            raise ValueError(
                "LLM_STRUCTURED_OUTPUT_PROTOCOL must be json_schema, guided_json, json_object, or off"
            )
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
    def occupation_code_path(self) -> Path:
        path = Path(self.occupation_code_csv_path)
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
