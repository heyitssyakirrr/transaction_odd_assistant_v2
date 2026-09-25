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

# Cuts generation short if the model keeps talking after its JSON object,
# instead of burning the rest of max_tokens on commentary. Deliberately NOT
# copied from a flat-schema stop list like "}\n{" or "}\n " -- those match
# an ordinary nested object's closing brace too (see the schemas in
# prompts.py, which nest arrays of objects) and would truncate a correct,
# in-progress response. These patterns only match a genuine blank line,
# which the system prompt explicitly tells the model not to
# produce inside a compact, single-line JSON object -- so seeing either one
# is itself a sign the model has moved past the JSON and started rambling.
# "```" is deliberately excluded: Qwen often opens with ```json, which would
# stop generation before any JSON and return empty content.
# This is a latency/cost optimisation, not the correctness mechanism -- see
# _parse_json_content for the part that actually guarantees a valid result.
_STOP_SEQUENCES = ["\n\n", "\r\n\r\n"]


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


class OpenAICompatibleClient:
    """Client for the team's configurable, OpenAI-compatible LLM loader.

    Handles the concerns a raw httpx call would otherwise leave to every
    caller: missing configuration surfaced as a clear error (rather than an
    obscure transport failure), and bounded retries with backoff for
    transient loader errors. The request shape defaults to what is confirmed
    to work against this org's loader (single `user` message, no
    `response_format`); see `_build_body` for how to opt back into
    `response_format` once/if the platform team confirms support for it.
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
        user_payload: dict[str, Any],
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
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
        try:
            return _parse_json_content(content)
        except LlmServiceError:
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
        user_payload: dict[str, Any],
        *,
        use_response_format: bool,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> dict[str, Any]:
        # use_response_format opts into schema-guided decoding. A bare
        # {"type": "json_schema"} with no nested "json_schema" object is NOT
        # a valid response_format and is silently ignored/no-ops on most
        # OpenAI-compatible loaders (confirmed against this org's loader:
        # sending it produced unconstrained, occasionally malformed JSON,
        # identical to sending nothing at all). The real schema MUST be
        # nested under json_schema.schema, generated from the same Pydantic
        # model that will validate the response (see analysis_service._ask),
        # so drift between the prompt's schema and the enforced schema is
        # impossible by construction.
        combined_prompt = f"{system_prompt}\n\n{json.dumps(user_payload, ensure_ascii=False, default=str)}"
        body: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [{"role": "user", "content": combined_prompt}],
            "temperature": 0,
            "max_tokens": self._settings.max_response_tokens,
            "stream": False,
            "stop": _STOP_SEQUENCES,
        }
        if self._settings.llm_frequency_penalty:
            body["frequency_penalty"] = self._settings.llm_frequency_penalty
        if use_response_format and response_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": response_schema,
                    "strict": True,
                },
            }
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
        budget = self._settings.llm_prompt_token_budget
        logger.info(
            "LLM request prepared: request_bytes=%d estimated_prompt_tokens=%d prompt_budget_tokens=%s max_response_tokens=%d",
            request_bytes,
            estimated_tokens,
            budget if budget is not None else "unconfigured",
            self._settings.max_response_tokens,
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


def _strip_code_fence(text: str) -> str:
    """Remove a wrapping ``` or ```json fence, in case the model adds one
    despite being told not to. A no-op if there is no fence."""
    candidate = text.strip()
    if not (candidate.startswith("```") and candidate.endswith("```")):
        return candidate
    candidate = candidate[3:]
    if candidate.lower().startswith("json"):
        candidate = candidate[4:]
    return candidate.rsplit("```", 1)[0].strip()


def _strip_trailing_commas(text: str) -> str:
    """Remove a comma immediately before a closing ] or } -- a common,
    otherwise-harmless slip that breaks strict JSON parsing."""
    return re.sub(r",(\s*[\]}])", r"\1", text)


def _extract_first_balanced_object(text: str) -> str | None:
    """Return the first complete, depth-balanced {...} span in text, or
    None if there isn't one.

    Tracks brace depth and string/escape state character by character, so
    it correctly finds the outermost object's true closing brace no matter
    how deeply nested the schema is underneath it -- unlike a fixed stop
    string, which cannot tell an inner object's closing brace from the
    outer one. Any text before or after this span (a stray greeting, a
    second object the model added by mistake) is discarded by construction,
    since only the first depth-zero span is returned.
    """
    depth = 0
    start: int | None = None
    in_string = False
    escape_next = False

    for index, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
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
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : index + 1]
    return None


def _close_unterminated_json(text: str) -> str:
    """Best-effort close any object/array left open by a response cut short
    mid-generation (e.g. by hitting max_tokens), by appending the correct
    closing characters in the correct order. A no-op if nothing is open.
    """
    stack: list[str] = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack and stack[-1] == char:
            stack.pop()

    if not stack:
        return text
    return text.rstrip().rstrip(",") + "".join(reversed(stack))


def _parse_json_content(content: str | dict[str, Any]) -> dict[str, Any]:
    """Parse the model's JSON output defensively.

    Well-formed output (the common case) parses on the very first attempt
    at no extra cost. Only when that fails does this fall through
    progressively more tolerant candidates -- stripping a markdown fence,
    extracting the first depth-balanced {...} span (discarding any
    commentary the model added around it), stripping a stray trailing
    comma, and closing brackets left open by a truncated response -- so a
    response that is imperfect but recoverable is not thrown away, while
    one that never contained a valid JSON object still fails loudly, same
    as before.
    """
    if isinstance(content, dict):
        return content
    if not isinstance(content, str) or not content.strip():
        raise LlmServiceError("LLM response content was empty.")

    unfenced = _strip_code_fence(content)
    candidates = [content, unfenced]

    extracted = _extract_first_balanced_object(unfenced)
    if extracted:
        candidates.append(extracted)

    for candidate in list(candidates):
        repaired = _strip_trailing_commas(_close_unterminated_json(candidate))
        if repaired not in candidates:
            candidates.append(repaired)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise LlmServiceError("LLM response content was not valid JSON.")


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