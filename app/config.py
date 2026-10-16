from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent # change based on where u put the file
_DEFAULT_LLM_STOP_SEQUENCES = ("<|im_end|>", "<|endoftext|>", "assistant:", "system:", "user:")


def _optional_float(name: str, default: str) -> float | None:
    """A number from .env; blank means "do not send this parameter"."""
    value = os.getenv(name, default).strip()
    return float(value) if value else None


def _stop_sequences(raw: str) -> tuple[str, ...]:
    """Comma-separated stop sequences; blank uses the Qwen defaults, "none" sends none."""
    if raw.strip().lower() == "none":
        return ()
    return tuple(value.strip() for value in raw.split(",") if value.strip()) or _DEFAULT_LLM_STOP_SEQUENCES


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
    # Sent as the X-PBAI-Helper-Id header when set (same name and header as the
    # Intelligent Document Extraction System).
    helper_id: str = os.getenv("HELPER_ID", "").strip()
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
    # Keep cookies the LLM server sets? Off by default: the app authenticates with the API key
    # header, and a load balancer's sticky-session cookie pins every later request to one
    # server, which then answers calls 1-3 one after another (seen in the 2026-10-12 logs).
    llm_keep_cookies: bool = os.getenv("LLM_KEEP_COOKIES", "false").lower() == "true"
    # Most LLM calls in flight at once, across all accounts. Each account sends up to 3 at once
    # (calls 1-3), so set it to BATCH_ACCOUNT_CONCURRENCY x 3; less makes accounts wait in the app.
    llm_concurrency: int = int(os.getenv("LLM_CONCURRENCY", "9"))
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
    # Multi-file upload: files wait in the application and run one account at a time,
    # so a batch never adds LLM calls beyond LLM_CONCURRENCY.
    batch_max_files: int = int(os.getenv("BATCH_MAX_FILES", "20"))
    # Finished batches kept in memory for the results page (cleared on restart).
    batch_keep_recent: int = int(os.getenv("BATCH_KEEP_RECENT", "20"))
    # Accounts of a batch analysed at the same time (one per LLM pod keeps every pod busy).
    batch_account_concurrency: int = int(os.getenv("BATCH_ACCOUNT_CONCURRENCY", "3"))
    customer_info_parquet_path: str = os.getenv(
        "CUSTOMER_INFO_PARQUET_PATH",
        str(BASE_DIR / "data" / "customer_info.parquet"), 
    )
    # Server folder of account Parquet files (one account per file, same 15 columns as the
    # CSV) for the "Read from folder" option; blank hides the option. Staff cannot change it.
    account_folder_path: str = os.getenv("ACCOUNT_FOLDER_PATH", "").strip()
    # CITIZEN_CD value in customer_info that means Malaysian, for the citizenship filter.
    malaysian_citizen_cd: str = os.getenv("MALAYSIAN_CITIZEN_CD", "MY").strip()
    # The risk the bank's rule-based program gave every account in the folder, written to
    # the batch results CSV to compare with ours. Never sent to the LLM. Blank leaves it out.
    folder_expected_risk: str = os.getenv("FOLDER_EXPECTED_RISK", "high").strip().lower()
    # One CSV per batch, a row per finished account: risk found, expected risk, report link.
    batch_results_directory: str = os.getenv("BATCH_RESULTS_DIRECTORY", str(BASE_DIR / "data" / "batches"))
    # CSV with columns CODE, D_CUST_OCCUPAT -- maintained by compliance,
    # loaded and cached at analysis time (see app/core/reference_data.py).
    occupation_code_csv_path: str = os.getenv(
        "OCCUPATION_CODE_CSV_PATH",
        str(BASE_DIR / "data" / "occupation_codes.csv"),
    )
    # Compatibility default for callers outside the account-context workflow.
    max_response_tokens: int = int(os.getenv("MAX_RESPONSE_TOKENS", "700"))
    # Four LLM calls per account. Calls 1-3 run concurrently through the shared work
    # queue: change in activity and amounts + money in and out, and activity after
    # inactivity + burst and gaps (both use the transaction cap below), and the
    # profile (its own cap). Call 4, the overall summary, runs after them.
    llm_transaction_max_response_tokens: int = int(os.getenv("LLM_TRANSACTION_MAX_RESPONSE_TOKENS", "500"))
    llm_profile_context_max_response_tokens: int = int(os.getenv("LLM_PROFILE_CONTEXT_MAX_RESPONSE_TOKENS", "320"))
    # Call 4: the overall summary and risk decision, run after calls 1-3 finish.
    llm_summary_max_response_tokens: int = int(os.getenv("LLM_SUMMARY_MAX_RESPONSE_TOKENS", "450"))
    # Activity after an inactive period. The run is computed from the six rows (see
    # app/core/monthly_facts.py) and given to the timing LLM call as a fact. It needs
    # at least this many consecutive zero-transaction months before a month with
    # transactions. This is an internal review rule, not a bank policy.
    dormancy_min_zero_months: int = int(os.getenv("DORMANCY_MIN_ZERO_MONTHS", "3"))
    # Every account file must hold exactly this many consecutive months (one row each).
    review_months: int = int(os.getenv("REVIEW_MONTHS", "12"))
    # Burst and gaps: a one-transaction month this many days or more after the previous
    # transaction is listed to the LLM as a long gap. Context only, never a risk level.
    review_long_gap_days: int = int(os.getenv("REVIEW_LONG_GAP_DAYS", "60"))
    # Screening for call 1: the code finds unusual transactions and layering months against
    # each account's OWN usual level (see app/core/monthly_facts.py); the AI describes the
    # evidence, and the summary judges it against the occupation and decides the risk.
    # Owned by compliance: change them here, never in code.
    # Unusual transaction: at least this many times the account's usual largest single transaction.
    review_unusual_times_usual: float = float(os.getenv("REVIEW_UNUSUAL_TIMES_USUAL", "2"))
    # The smallest amount worth reviewing: a single transaction (or the money that passed
    # through a month) below it is never a signal. Accounts with too little history to have a
    # usual level are screened on this amount alone.
    review_min_amount: float = float(os.getenv("REVIEW_MIN_AMOUNT", "2000"))
    # Layering: money in and money out within this percentage of each other (of the larger
    # side), in a month at least REVIEW_LAYERING_TIMES_USUAL_MONTH times the usual month.
    review_layering_max_difference_pct: float = float(os.getenv("REVIEW_LAYERING_MAX_DIFFERENCE_PCT", "15"))
    review_layering_times_usual_month: float = float(os.getenv("REVIEW_LAYERING_TIMES_USUAL_MONTH", "2"))
    # Retained for backwards-compatible .env files; no longer used by the
    # account-context workflow.
    llm_timeline_max_response_tokens: int = int(os.getenv("LLM_TIMELINE_MAX_RESPONSE_TOKENS", "260"))
    llm_flow_max_response_tokens: int = int(os.getenv("LLM_FLOW_MAX_RESPONSE_TOKENS", "260"))
    llm_profile_max_response_tokens: int = int(os.getenv("LLM_PROFILE_MAX_RESPONSE_TOKENS", "220"))
    llm_synthesis_max_response_tokens: int = int(os.getenv("LLM_SYNTHESIS_MAX_RESPONSE_TOKENS", "260"))
    # Name of the output-limit field in the request body. The Foundry gateway takes
    # max_tokens (per the platform team); set max_completion_tokens if it ever asks for that.
    llm_max_tokens_param: str = os.getenv("LLM_MAX_TOKENS_PARAM", "max_tokens").strip()
    # Highest output limit any call may request; 0 = no cap. 700 protects Qwen from long
    # runaway answers. Foundry (gpt-6-luna) needs room for its hidden reasoning: set 0 or 16000.
    llm_output_token_cap: int = int(os.getenv("LLM_OUTPUT_TOKEN_CAP", "700"))
    # Reasoning models (gpt-6-luna) think before answering, and those hidden tokens count
    # against the output limit. This many tokens are added to every call's own limit
    # (LLM_TRANSACTION_MAX_RESPONSE_TOKENS etc.) so the answer is not cut off. 0 for Qwen.
    llm_reasoning_extra_tokens: int = int(os.getenv("LLM_REASONING_EXTRA_TOKENS", "0"))
    # Sent as reasoning_effort (low / medium / high) when set; blank = the model's default.
    llm_reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
    # Qwen's low-variance sampling defaults. Leave blank in .env to not send the
    # parameter at all: Foundry reasoning models reject temperature and top_p with a 400.
    llm_temperature: float | None = _optional_float("LLM_TEMPERATURE", "0")
    llm_top_p: float | None = _optional_float("LLM_TOP_P", "0.001")
    # This is a vLLM/Qwen extension rather than a portable OpenAI parameter.
    # Leave disabled unless the bank loader is confirmed to accept it.
    llm_repetition_penalty: float = float(os.getenv("LLM_REPETITION_PENALTY", "0"))
    # The bank loader currently ignores both guided_json and response_format.
    # Keep this off until platform support is confirmed; flat contracts plus
    # bounded mechanical recovery provide the compatibility path.
    llm_structured_output_protocol: str = os.getenv("LLM_STRUCTURED_OUTPUT_PROTOCOL", "off").lower()
    llm_json_repair_enabled: bool = os.getenv("LLM_JSON_REPAIR_ENABLED", "true").lower() == "true"
    # A malformed but complete response may succeed after one clean retry.
    # Truncated (`finish_reason=length`) responses never retry because the
    # same prompt and output cap would be expected to fail again.
    llm_format_retry_enabled: bool = os.getenv("LLM_FORMAT_RETRY_ENABLED", "true").lower() == "true"
    # Use commas, not pipes: Qwen's native special tokens themselves contain
    # pipes. Literal assistant: prevents the loader from starting a duplicate
    # assistant turn after a complete object. "none" sends no stop sequences
    # (Foundry reasoning models reject "stop").
    llm_stop_sequences: tuple[str, ...] = _stop_sequences(os.getenv("LLM_STOP_SEQUENCES", ""))
    # Every answer is one flat JSON object, so its first } is its end. Stopping there
    # stops Qwen from repeating the object or explaining it until the token limit, which
    # saves time. The loader drops the stop text, so the client adds the } back.
    # Foundry: false (it is sent as a stop sequence, which reasoning models reject).
    llm_stop_after_json_object: bool = os.getenv("LLM_STOP_AFTER_JSON_OBJECT", "true").lower() == "true"
    report_directory: str = os.getenv("REPORT_DIRECTORY", str(BASE_DIR / "data" / "reports"))
    # Application logs are persisted separately from generated staff reports.
    # Mount this directory on persistent storage in OpenShift when logs must
    # survive a pod replacement.
    log_directory: str = os.getenv("LOG_DIRECTORY", str(BASE_DIR / "data" / "logs"))
    log_retention_days: int = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    require_human_review: bool = os.getenv("REQUIRE_HUMAN_REVIEW", "true").lower() == "true"

    def __post_init__(self) -> None:
        positive_values = {
            "LLM_TIMEOUT_SECONDS": self.llm_timeout_seconds,
            "LLM_CONCURRENCY": self.llm_concurrency,
            "LLM_QUEUE_MAXSIZE": self.llm_queue_maxsize,
            "LLM_QUEUE_ENQUEUE_TIMEOUT_SECONDS": self.llm_queue_enqueue_timeout_seconds,
            "BATCH_MAX_FILES": self.batch_max_files,
            "BATCH_KEEP_RECENT": self.batch_keep_recent,
            "BATCH_ACCOUNT_CONCURRENCY": self.batch_account_concurrency,
            "MAX_RESPONSE_TOKENS": self.max_response_tokens,
            "LLM_TRANSACTION_MAX_RESPONSE_TOKENS": self.llm_transaction_max_response_tokens,
            "LLM_PROFILE_CONTEXT_MAX_RESPONSE_TOKENS": self.llm_profile_context_max_response_tokens,
            "LLM_SUMMARY_MAX_RESPONSE_TOKENS": self.llm_summary_max_response_tokens,
            "DORMANCY_MIN_ZERO_MONTHS": self.dormancy_min_zero_months,
            "REVIEW_MONTHS": self.review_months,
            "REVIEW_LONG_GAP_DAYS": self.review_long_gap_days,
            "REVIEW_UNUSUAL_TIMES_USUAL": self.review_unusual_times_usual,
            "REVIEW_MIN_AMOUNT": self.review_min_amount,
            "REVIEW_LAYERING_MAX_DIFFERENCE_PCT": self.review_layering_max_difference_pct,
            "REVIEW_LAYERING_TIMES_USUAL_MONTH": self.review_layering_times_usual_month,
            "LLM_TIMELINE_MAX_RESPONSE_TOKENS": self.llm_timeline_max_response_tokens,
            "LLM_FLOW_MAX_RESPONSE_TOKENS": self.llm_flow_max_response_tokens,
            "LLM_PROFILE_MAX_RESPONSE_TOKENS": self.llm_profile_max_response_tokens,
            "LLM_SYNTHESIS_MAX_RESPONSE_TOKENS": self.llm_synthesis_max_response_tokens,
            "LLM_LOG_RAW_RESPONSE_MAX_CHARS": self.llm_log_raw_response_max_chars,
            "LOG_RETENTION_DAYS": self.log_retention_days,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be greater than zero: {', '.join(invalid)}")
        if self.llm_context_window_tokens < 0:
            raise ValueError("LLM_CONTEXT_WINDOW_TOKENS must be zero or greater")
        if self.llm_context_safety_margin_tokens < 0:
            raise ValueError("LLM_CONTEXT_SAFETY_MARGIN_TOKENS must be zero or greater")
        if self.llm_temperature is not None and not 0 <= self.llm_temperature <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2, or blank to not send it")
        if self.llm_top_p is not None and not 0 < self.llm_top_p <= 1:
            raise ValueError("LLM_TOP_P must be greater than 0 and no greater than 1, or blank to not send it")
        if self.llm_max_tokens_param not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("LLM_MAX_TOKENS_PARAM must be max_tokens or max_completion_tokens")
        if self.llm_output_token_cap < 0:
            raise ValueError("LLM_OUTPUT_TOKEN_CAP must be zero (no cap) or greater")
        if self.llm_reasoning_extra_tokens < 0:
            raise ValueError("LLM_REASONING_EXTRA_TOKENS must be zero or greater")
        if self.llm_reasoning_effort not in {"", "minimal", "low", "medium", "high"}:
            raise ValueError("LLM_REASONING_EFFORT must be minimal, low, medium, high or blank")
        if self.folder_expected_risk not in {"", "low", "medium", "high"}:
            raise ValueError("FOLDER_EXPECTED_RISK must be low, medium, high or blank")
        if not 1 <= self.review_months <= 24:
            raise ValueError("REVIEW_MONTHS must be between 1 and 24")
        if not 0 < self.review_layering_max_difference_pct < 100:
            raise ValueError("REVIEW_LAYERING_MAX_DIFFERENCE_PCT must be between 0 and 100")
        if self.llm_repetition_penalty < 0:
            raise ValueError("LLM_REPETITION_PENALTY must not be negative")
        if self.llm_structured_output_protocol not in {"json_schema", "guided_json", "json_object", "off"}:
            raise ValueError(
                "LLM_STRUCTURED_OUTPUT_PROTOCOL must be json_schema, guided_json, json_object, or off"
            )
        if self.llm_context_window_tokens and self.llm_context_window_tokens <= self.max_response_tokens + self.llm_context_safety_margin_tokens:
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
    def log_directory_path(self) -> Path:
        path = Path(self.log_directory)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def customer_info_path(self) -> Path:
        path = Path(self.customer_info_parquet_path)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def llm_calls_needed(self) -> int:
        """LLM_CONCURRENCY at which no account's calls wait in the app: 3 calls per account at once."""
        return self.batch_account_concurrency * 3

    @property
    def account_folder(self) -> Path | None:
        if not self.account_folder_path:
            return None
        path = Path(self.account_folder_path)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def batch_results_path(self) -> Path:
        path = Path(self.batch_results_directory)
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