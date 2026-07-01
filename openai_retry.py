"""Resilient wrapper around OpenAI chat completions.

The CV parser hits OpenAI for text extraction, document classification and
vision parsing. Two distinct 429s can come back:

  * rate_limit_exceeded  — transient; the right response is backoff + retry.
  * insufficient_quota    — a billing problem; retrying cannot help, so we fail
                            fast with a clear, actionable message instead of
                            burning retries and time.

This helper retries transient 429s / timeouts / connection errors / 5xx with
exponential backoff + jitter (honoring Retry-After when present), and raises
``OpenAIQuotaExhausted`` immediately on insufficient_quota.
"""
from __future__ import annotations

import logging
import random
import time

import openai

logger = logging.getLogger("openai_retry")

# Transient error classes that are always safe to retry.
_RETRYABLE = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIQuotaExhausted(RuntimeError):
    """OpenAI returned insufficient_quota — needs credits / a higher tier."""


def _is_quota_error(e: Exception) -> bool:
    """True when a 429 is insufficient_quota (billing) rather than a rate limit."""
    if getattr(e, "code", None) == "insufficient_quota":
        return True
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("type") == "insufficient_quota":
            return True
    text = str(getattr(e, "message", "") or e).lower()
    return "insufficient_quota" in text or "exceeded your current quota" in text


def _sleep_seconds(e: Exception, attempt: int, base_delay: float) -> float:
    """Retry-After header if the server sent one, else exponential backoff + jitter."""
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            ra = resp.headers.get("retry-after")
            if ra:
                return min(float(ra), 30.0) + random.uniform(0, 0.5)
        except Exception:
            pass
    return min(base_delay * (2 ** (attempt - 1)), 30.0) + random.uniform(0, 1.0)


def chat_completion_with_retry(*, client=None, max_attempts: int = 5, base_delay: float = 2.0, **kwargs):
    """Call chat.completions.create with retry/backoff.

    Pass ``client=`` to use an explicit OpenAI() instance; omit it to use the
    module-level default client. All other kwargs are forwarded verbatim
    (model, messages, temperature, ...).
    """
    caller = client if client is not None else openai
    label = kwargs.get("model", "openai")
    attempt = 0
    while True:
        try:
            return caller.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            if _is_quota_error(e):
                logger.error("[openai_retry] insufficient_quota on %s — not retrying", label)
                raise OpenAIQuotaExhausted(
                    "OpenAI quota exhausted (insufficient_quota): add credits or raise the "
                    "account tier — retrying cannot help."
                ) from e
            attempt += 1
            if attempt >= max_attempts:
                logger.error("[openai_retry] 429 rate limit on %s — giving up after %d attempts", label, attempt)
                raise
            delay = _sleep_seconds(e, attempt, base_delay)
            logger.warning("[openai_retry] 429 rate limit on %s; retry %d/%d in %.1fs", label, attempt, max_attempts - 1, delay)
            time.sleep(delay)
        except _RETRYABLE as e:
            attempt += 1
            if attempt >= max_attempts:
                logger.error("[openai_retry] %s on %s — giving up after %d attempts", type(e).__name__, label, attempt)
                raise
            delay = _sleep_seconds(e, attempt, base_delay)
            logger.warning("[openai_retry] %s on %s; retry %d/%d in %.1fs", type(e).__name__, label, attempt, max_attempts - 1, delay)
            time.sleep(delay)
