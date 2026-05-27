"""
client.py — Generic async LLM client for OpenAI-compatible REST APIs.

Currently supports OpenRouter. Other providers can be added to _PROVIDER_DEFAULTS.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key":  "OPENROUTER_API_KEY",
    },
}

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    response_text: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    finish_reason: Optional[str]
    latency_ms: int
    model_used: Optional[str] = None   # exact model ID returned by the API
    cost_usd: Optional[float] = None   # upstream inference cost in USD (OpenRouter)
    error: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


# ── Client ────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Async wrapper around an OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    provider    : "openrouter" (or any key in _PROVIDER_DEFAULTS)
    api_key     : explicit key; falls back to the provider's env var
    timeout     : per-request timeout in seconds
    max_retries : max retry attempts on transient errors
    """

    def __init__(
        self,
        provider: str = "openrouter",
        api_key: str | None = None,
        http_referer: str = "https://github.com/longitudinal-llm-study",
        site_name: str = "LLM Dataset Study",
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        defaults = _PROVIDER_DEFAULTS.get(provider)
        if defaults is None:
            raise ValueError(
                f"Unknown provider {provider!r}. "
                f"Add it to _PROVIDER_DEFAULTS or use: {list(_PROVIDER_DEFAULTS)}"
            )

        self._provider   = provider
        self._api_key    = api_key or os.environ[defaults["env_key"]]
        self._base_url   = defaults["base_url"]
        self._referer    = http_referer
        self._site_name  = site_name
        self._timeout    = timeout
        self._max_retries = max_retries

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(timeout),
        )

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._provider == "openrouter":
            h["HTTP-Referer"] = self._referer
            h["X-Title"] = self._site_name
        return h

    async def chat(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Send one chat completion request and return an LLMResponse.

        Retries on transient failures with exponential back-off.
        Never raises — errors are captured in LLMResponse.error.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if extra_params:
            body.update(extra_params)

        t0 = time.monotonic()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=1, min=2, max=60),
                retry=retry_if_exception_type(
                    (httpx.TransportError, _RetryableHTTPError)
                ),
                reraise=True,
            ):
                with attempt:
                    resp = await self._http.post(
                        "/chat/completions", content=json.dumps(body)
                    )
                    if resp.status_code in _RETRYABLE_STATUS:
                        raise _RetryableHTTPError(resp.status_code, resp.text)
                    resp.raise_for_status()

            latency_ms = int((time.monotonic() - t0) * 1000)
            data   = resp.json()
            choice = data["choices"][0]
            usage  = data.get("usage", {})

            text = choice["message"]["content"]

            # Append OpenRouter web-search citations if present
            if self._provider == "openrouter":
                annotations = choice["message"].get("annotations") or []
                sources = []
                for ann in annotations:
                    if ann.get("type") == "url_citation":
                        uc  = ann.get("url_citation", {})
                        url = uc.get("url")
                        if url:
                            title = uc.get("title", "")
                            sources.append(f"[{title}]({url})" if title else url)
                if sources:
                    text = text + "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

            return LLMResponse(
                response_text=text,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                finish_reason=choice.get("finish_reason"),
                latency_ms=latency_ms,
                model_used=data.get("model"),
                cost_usd=usage.get("cost"),
                raw=data,
            )

        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            return LLMResponse(
                response_text=None,
                input_tokens=None,
                output_tokens=None,
                finish_reason=None,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


# ── Internal ──────────────────────────────────────────────────────────────────

class _RetryableHTTPError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status


# ── Model self-identification ─────────────────────────────────────────────────

async def identify_model(client: "LLMClient", model_id: str) -> str | None:
    """
    Ask the model what version it is. Returns the self-reported name, or None
    on failure. Intended to be called once per model per wave, not per prompt.
    """
    resp = await client.chat(
        model=model_id,
        prompt=(
            "What is the exact model version you are running on? "
            "Reply in one short sentence, e.g. 'I am GPT-5.5' or 'I am Claude 4 Sonnet'. "
            "Do not add any other text."
        ),
        temperature=0.0,
        max_tokens=64,
    )
    if resp.error or not resp.response_text:
        return None
    return resp.response_text.strip()


# ── OpenRouter model catalogue ────────────────────────────────────────────────

async def list_openrouter_models(api_key: str | None = None) -> list[dict[str, Any]]:
    key = api_key or os.environ["OPENROUTER_API_KEY"]
    async with httpx.AsyncClient(timeout=30) as hc:
        resp = await hc.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
