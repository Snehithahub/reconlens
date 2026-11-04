"""Small OpenAI-compatible chat client (Groq, Gemini, OpenAI, Azure OpenAI) returning parsed JSON."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from reconlens.config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class JSONLLM(Protocol):
    def complete_json(self, system: str, user: str, *, max_tokens: int = 1500) -> dict[str, Any]: ...


def _retryable(e: BaseException) -> bool:
    if isinstance(e, httpx.TransportError):
        return True
    return isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (408, 429, 500, 502, 503, 504)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating ```json fences and leading prose."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        val = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise LLMError("model did not return JSON") from None
        try:
            val = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise LLMError("model returned malformed JSON") from e
    if not isinstance(val, dict):
        raise LLMError("model returned JSON that is not an object")
    return val


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str, *, provider: str = "custom",
                 api_version: str = "", timeout_s: float = 30.0, max_retries: int = 3,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        headers = {"api-key": api_key} if provider == "azure" else {"Authorization": f"Bearer {api_key}"}
        params = {"api-version": api_version} if provider == "azure" else {}
        self._http = httpx.Client(base_url=base_url, headers=headers, params=params,
                                  timeout=httpx.Timeout(timeout_s), transport=transport)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(self.max_retries),
               wait=wait_exponential(multiplier=0.5, max=8), reraise=True)
        def call() -> dict[str, Any]:
            r = self._http.post("/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()

        return call()

    def complete_json(self, system: str, user: str, *, max_tokens: int = 1500) -> dict[str, Any]:
        payload = {
            "model": self.model, "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            data = self._post(payload)
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}") from e
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("unexpected LLM response shape") from e
        return parse_json_object(content or "")


def build_client(settings: Settings) -> OpenAICompatClient | None:
    resolved = settings.resolved_llm()
    if resolved is None:
        return None
    base, model = resolved
    return OpenAICompatClient(base, model, settings.llm_api_key or "", provider=settings.llm_provider,
                              api_version=settings.llm_api_version, timeout_s=settings.llm_timeout_s,
                              max_retries=settings.llm_max_retries)
