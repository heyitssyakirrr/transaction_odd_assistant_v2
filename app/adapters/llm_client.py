from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger("app.llm_client")

# Transient failures worth a retry: connection issues, timeouts, and the
# status codes an upstream loader typically returns while overloaded or
# warming up. Anything else (4xx client errors) is not retried, since
# retrying a malformed request just burns one of the loader's few slots.
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 1
_BACKOFF_BASE_SECONDS = 0.75
_QWEN_ASSESSMENT_TOKEN_CAP = 700


class LlmServiceError(RuntimeError):
    """A safe, classified failure returned by the upstream LLM service."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        upstream_request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.upstream_request_id = upstream_request_id


class LlmContextWindowError(LlmServiceError):
    """The request cannot fit within the configured or reported model context."""


class LlmOutputFormatError(LlmServiceError):
    """The model answer cannot be safely normalised into one JSON object."""


class OpenAICompatibleClient:
    """Compatibility client for a loader that does not enforce JSON schemas.

    It permits only mechanical punctuation recovery. The downstream Pydantic
    schema and source-citation checks remain the authority for acceptance.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=settings.llm_timeout_seconds,
            limits=httpx.Limits(
                max_connections=settings.llm_concurrency,
                max_keepalive_connections=settings.llm_concurrency,
            ),
            verify=False
        )

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any] | str,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        max_response_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not self._settings.llm_base_url:
            raise LlmServiceError(
                "LLM_BASE_URL is not configured. Copy .env.example to .env in the "
                "project root and set LLM_BASE_URL to the internal loader's address."
            )

        headers = self._build_headers()
        body = self._build_body(
            system_prompt,
            user_payload,
            use_response_format=True,
            response_schema=response_schema,
            schema_name=schema_name,
            max_response_tokens=max_response_tokens,
        )
        self._validate_context_budget(body)
        payload = await self._post_with_retries(body, headers)

        choice = (payload.get("choices") or [{}])[0]
        logger.info(
            "LLM completion: finish_reason=%s completion_tokens=%s",
            choice.get("finish_reason"),
            (payload.get("usage") or {}).get("completion_tokens"),
        )
        content = _extract_content(payload)
        self._log_raw_response(content)
        if choice.get("finish_reason") == "length":
            raise LlmOutputFormatError(
                "LLM output reached MAX_RESPONSE_TOKENS and was rejected; it was not used as an assessment."
            )
        try:
            return _parse_json_content(content, repair=self._settings.llm_json_repair_enabled)
        except LlmOutputFormatError:
            logger.warning(
                "LLM returned invalid JSON: response_chars=%d", len(content) if isinstance(content, str) else 0
            )
            raise

    async def close(self) -> None:
        await self._client.aclose()

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.llm_api_key:
            value = self._settings.llm_api_key
            if self._settings.llm_api_key_header.lower() == "authorization":
                value = f"Bearer {value}"
            headers[self._settings.llm_api_key_header] = value
        return headers

    def _build_body(
        self,
        system_prompt: str,
        user_payload: dict[str, Any] | str,
        *,
        use_response_format: bool,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        max_response_tokens: int | None = None,
    ) -> dict[str, Any]:
        requested_tokens = max_response_tokens or self._settings.max_response_tokens
        output_tokens = min(requested_tokens, _QWEN_ASSESSMENT_TOKEN_CAP)
        if requested_tokens > _QWEN_ASSESSMENT_TOKEN_CAP:
            logger.warning(
                "LLM response-token request clamped for Qwen assessment: requested=%d cap=%d",
                requested_tokens, _QWEN_ASSESSMENT_TOKEN_CAP,
            )
        body: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=False, default=str)},
            ],
            "temperature": self._settings.llm_temperature,
            "top_p": self._settings.llm_top_p,
            "max_tokens": output_tokens,
            "stream": False,
        }
        if self._settings.llm_repetition_penalty:
            body["repetition_penalty"] = self._settings.llm_repetition_penalty
        if self._settings.llm_stop_sequences:
            body["stop"] = list(self._settings.llm_stop_sequences)
        if use_response_format and response_schema:
            protocol = self._settings.llm_structured_output_protocol
            if protocol == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": response_schema,
                    },
                }
            elif protocol == "guided_json":
                # vLLM's legacy OpenAI-compatible structured-output field.
                body["guided_json"] = response_schema
            elif protocol == "json_object":
                body["response_format"] = {"type": "json_object"}
        return body

    async def _post_with_retries(self, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            started_at = asyncio.get_running_loop().time()
            try:
                response = await self._client.post(self._settings.chat_url, headers=headers, json=body)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "LLM HTTP request failed: attempt=%d/%d error_type=%s",
                    attempt,
                    _MAX_ATTEMPTS,
                    type(exc).__name__,
                )
                if attempt == _MAX_ATTEMPTS:
                    break
                await self._sleep_before_retry(attempt, reason=type(exc).__name__)
                continue

            duration_ms = int((asyncio.get_running_loop().time() - started_at) * 1000)
            request_id = _upstream_request_id(response)
            logger.info(
                "LLM HTTP response: status=%d attempt=%d/%d duration_ms=%d upstream_request_id=%s",
                response.status_code,
                attempt,
                _MAX_ATTEMPTS,
                duration_ms,
                request_id or "-",
            )

            if _is_context_window_response(response):
                diagnostic = _safe_error_message(response)
                logger.error(
                    "LLM rejected prompt for context-window limit: status=%d upstream_request_id=%s detail=%s",
                    response.status_code,
                    request_id or "-",
                    diagnostic,
                )
                raise LlmContextWindowError(
                    "The LLM rejected this request because it exceeds its context window. "
                    "Reduce CHUNK_SIZE or configure the correct context-window limit.",
                    status_code=response.status_code,
                    upstream_request_id=request_id,
                )

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_ATTEMPTS:
                logger.warning(
                    "LLM loader returned %s on attempt %s/%s; retrying. upstream_request_id=%s detail=%s",
                    response.status_code,
                    attempt,
                    _MAX_ATTEMPTS,
                    request_id or "-",
                    _safe_error_message(response),
                )
                await self._sleep_before_retry(attempt, reason=f"HTTP {response.status_code}")
                continue

            try:
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.error(
                    "LLM request failed permanently: status=%d upstream_request_id=%s detail=%s",
                    response.status_code,
                    request_id or "-",
                    _safe_error_message(response),
                )
                break

        raise LlmServiceError(
            f"LLM service request failed after {_MAX_ATTEMPTS} attempts: {last_error}",
            status_code=getattr(getattr(last_error, "response", None), "status_code", None),
        )

    def _validate_context_budget(self, body: dict[str, Any]) -> None:
        serialized = json.dumps(body["messages"], ensure_ascii=False, separators=(",", ":"))
        request_bytes = len(serialized.encode("utf-8"))
        estimated_tokens = _estimate_tokens(serialized)
        budget = None
        if self._settings.llm_context_window_tokens:
            budget = (
                self._settings.llm_context_window_tokens
                - body["max_tokens"]
                - self._settings.llm_context_safety_margin_tokens
            )
        logger.info(
            "LLM request prepared: request_bytes=%d estimated_prompt_tokens=%d structured_protocol=%s prompt_budget_tokens=%s max_response_tokens=%d",
            request_bytes,
            estimated_tokens,
            self._settings.llm_structured_output_protocol,
            budget if budget is not None else "unconfigured",
            body["max_tokens"],
        )
        if budget is not None and estimated_tokens > budget:
            logger.error(
                "LLM request rejected before send for configured context budget: estimated_prompt_tokens=%d prompt_budget_tokens=%d",
                estimated_tokens,
                budget,
            )
            raise LlmContextWindowError(
                "The request is estimated to exceed the configured LLM context window. "
                "Reduce CHUNK_SIZE or increase the confirmed model context-window setting.",
            )

    def _log_raw_response(self, content: str | dict[str, Any]) -> None:
        """Log model content only when explicitly enabled for diagnostics."""
        if not self._settings.llm_log_raw_response:
            return
        raw = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        limit = self._settings.llm_log_raw_response_max_chars
        logger.info(
            "LLM raw response: chars=%d truncated=%s content=%s",
            len(raw),
            len(raw) > limit,
            raw[:limit],
        )

    @staticmethod
    async def _sleep_before_retry(attempt: int, *, reason: str) -> None:
        delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
        logger.debug("Backing off %.2fs before retry (%s).", delay, reason)
        await asyncio.sleep(delay)


def _extract_content(payload: dict[str, Any]) -> str | dict[str, Any]:
    if payload.get("choices"):
        content = payload["choices"][0].get("message", {}).get("content")
        if content is not None:
            return content
    if payload.get("text") is not None:
        return payload["text"]
    raise LlmServiceError("Unrecognised LLM response: expected choices[0].message.content or text")


def _parse_json_content(content: str | dict[str, Any], *, repair: bool = False) -> dict[str, Any]:
    """Parse one model object, with optional bounded punctuation recovery.

    Recovery removes only an outer code fence/trailing comma and appends closing
    quote/brackets for one unfinished object. It never supplies a missing key,
    evidence ID, value, or finding. Pydantic validation still rejects incomplete
    business content immediately afterwards.
    """
    if isinstance(content, dict):
        return content
    if not isinstance(content, str) or not content.strip():
        raise LlmOutputFormatError("LLM response content was empty.")

    candidates = [content.strip()]
    fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    if repair:
        candidates.extend(_top_level_json_objects(content))
        recovered = _recover_json_object(content)
        if recovered:
            candidates.append(recovered)

    for candidate in reversed(_unique(candidates)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LlmOutputFormatError("LLM response was not a recoverable JSON object.")


def _top_level_json_objects(text: str) -> list[str]:
    """Return complete top-level JSON object candidates without parsing prose."""
    candidates: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:index + 1])
                start = None
    return candidates


def _recover_json_object(text: str) -> str | None:
    """Make syntactic complements only; never invent semantic JSON content."""
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:].strip()
    fragment = re.sub(r"\s*```\s*$", "", fragment)
    stack: list[str] = []
    in_string = False
    escaped = False
    result: list[str] = []
    for char in fragment:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            result.append(char)
            escaped = True
            continue
        if char == '"':
            result.append(char)
            in_string = not in_string
            continue
        if not in_string and char in "{[":
            stack.append("}" if char == "{" else "]")
        elif not in_string and char in "}]":
            if not stack:
                return None
            # Example: `{"items":[1,2,}` is missing only `]`. Close that
            # open array before accepting the model's following object close.
            while stack and char != stack[-1]:
                result.append(stack.pop())
            if not stack:
                return None
            stack.pop()
        result.append(char)
    if in_string:
        result.append('"')
    recovered = "".join(result).rstrip() + "".join(reversed(stack))
    return _remove_trailing_commas(recovered)


def _remove_trailing_commas(text: str) -> str:
    """Remove a comma only when it immediately precedes a structural close."""
    result: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if escaped:
            result.append(char)
            escaped = False
        elif char == "\\" and in_string:
            result.append(char)
            escaped = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif char == "," and not in_string:
            next_index = index + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index < len(text) and text[next_index] in "}]":
                index += 1
                continue
            result.append(char)
        else:
            result.append(char)
        index += 1
    return "".join(result)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _estimate_tokens(text: str) -> int:
    """Conservative, tokenizer-agnostic estimate used only for a safety guard."""
    return (len(text) + 3) // 4


def _upstream_request_id(response: httpx.Response) -> str | None:
    for header in ("x-request-id", "x-correlation-id", "traceparent"):
        if value := response.headers.get(header):
            return value[:200]
    return None


def _is_context_window_response(response: httpx.Response) -> bool:
    if response.status_code not in {400, 413, 422}:
        return False
    detail = _safe_error_message(response).lower()
    indicators = ("context length", "context window", "too many tokens", "prompt too long", "maximum tokens")
    return any(indicator in detail for indicator in indicators)


def _safe_error_message(response: httpx.Response) -> str:
    """Return a bounded diagnostic without logging an arbitrary upstream body."""
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return "non-JSON upstream error body"
    if isinstance(payload, dict):
        error = payload.get("error", payload.get("detail", payload.get("message", "unknown upstream error")))
        if isinstance(error, dict):
            error = error.get("message", error.get("code", "unknown upstream error"))
        return str(error).replace("\n", " ")[:500]
    return "unstructured upstream error"
