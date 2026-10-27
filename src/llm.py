"""Shared Gemini client: persistent cache, global rate limit, backoff, call log.

Every LLM call in the project goes through `LLMClient`. Responses are cached in a
JSONL file keyed by (model, prompt_version, schema, prompt), so reruns are free and
deterministic. Real calls are appended to a call log so quota use can be tracked.
Tests inject a fake `transport` and never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel

from src import ROOT

# transport(model, prompt, schema, temperature, max_output_tokens) -> response text
Transport = Callable[[str, str, type[BaseModel] | None, float, int], str]

RETRYABLE_CODES = {429, 500, 502, 503, 504}
# Judge calls with thinking take up to ~5 min; without a timeout a stalled request hung for 8 h (27 Oct).
REQUEST_TIMEOUT_MS = 10 * 60 * 1000
# Free-tier daily quotas reset at midnight Pacific. PDT (UTC-7) holds until 1 Nov 2026.
QUOTA_DAY_TZ = timezone(timedelta(hours=-7))

_last_call_at: dict[str, float] = {}  # process-wide rate limiter state, per model


class QuotaExhausted(RuntimeError):
    """Daily quota used up, or 429s persisted after backoff. Stop and resume later."""


class CallLimitReached(RuntimeError):
    """The `max_calls` budget for this client (dry-run limit) was reached."""


def gemini_transport(model: str, prompt: str, schema: type[BaseModel] | None,
                     temperature: float, max_output_tokens: int) -> str:
    """Call the Gemini API once and return the response text."""
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json" if schema else "text/plain",
        response_schema=schema,
    )
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    return response.text or ""


def _error_code(exc: BaseException) -> int | None:
    """HTTP-like status code of an API error, if it has one."""
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _is_daily_quota(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "perday" in msg or "per day" in msg or "requests_per_day" in msg


class LLMClient:
    """Cached, rate-limited, retrying LLM client for one model."""

    def __init__(
        self,
        model: str,
        cache_path: Path | str = ROOT / "artifacts" / "llm_cache.jsonl",
        call_log_path: Path | str = ROOT / "artifacts" / "llm_calls.jsonl",
        rpm: float = 10,
        temperature: float = 0.0,
        max_output_tokens: int = 400,
        max_calls: int | None = None,
        max_attempts: int = 5,
        base_delay: float = 2.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.cache_path = Path(cache_path)
        self.call_log_path = Path(call_log_path)
        self.min_interval = 60.0 / rpm
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_calls = max_calls
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.transport = transport or gemini_transport
        self.sleep = sleep
        self.clock = clock
        self.calls_made = 0  # real API calls by this instance (retries included)
        self.cache_hits = 0
        self._cache = self._load_cache()

    # ---- cache -------------------------------------------------------------
    def _load_cache(self) -> dict[str, str]:
        cache: dict[str, str] = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        cache[row["key"]] = row["response"]
        return cache

    def cache_key(self, prompt: str, prompt_version: str, schema: type[BaseModel] | None) -> str:
        payload = json.dumps(
            [self.model, prompt_version, schema.model_json_schema() if schema else None,
             self.temperature, self.max_output_tokens, prompt],
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _append_jsonl(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- calling -------------------------------------------------------------
    def _wait_for_rate_limit(self) -> None:
        last = _last_call_at.get(self.model)
        if last is not None:
            wait = self.min_interval - (self.clock() - last)
            if wait > 0:
                self.sleep(wait)
        _last_call_at[self.model] = self.clock()

    def _call_with_backoff(self, prompt: str, schema: type[BaseModel] | None, prompt_version: str) -> str:
        for attempt in range(1, self.max_attempts + 1):
            if self.max_calls is not None and self.calls_made >= self.max_calls:
                raise CallLimitReached(f"max_calls={self.max_calls} reached for {self.model}")
            self._wait_for_rate_limit()
            self.calls_made += 1
            try:
                text = self.transport(self.model, prompt, schema, self.temperature, self.max_output_tokens)
                self._log_call(prompt_version, "ok")
                return text
            except Exception as exc:  # noqa: BLE001 - classified below
                code = _error_code(exc)
                self._log_call(prompt_version, f"error {code}")
                if code == 429 and _is_daily_quota(exc):
                    raise QuotaExhausted(f"daily quota exhausted for {self.model}") from exc
                # Dropped connections and timeouts have no status code but are transient.
                if code not in RETRYABLE_CODES and not isinstance(exc, httpx.TransportError):
                    raise
                if attempt == self.max_attempts:
                    if code == 429:
                        raise QuotaExhausted(f"429 persisted after {attempt} attempts for {self.model}") from exc
                    raise
                delay = self.base_delay * (2 ** (attempt - 1)) * (0.5 + random.random())
                print(f"[llm] {self.model} error {code}, attempt {attempt}/{self.max_attempts}; retry in {delay:.1f}s")
                self.sleep(delay)
        raise AssertionError("unreachable")

    def _log_call(self, prompt_version: str, status: str) -> None:
        self._append_jsonl(self.call_log_path, {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": self.model, "prompt_version": prompt_version, "status": status,
        })

    def generate(self, prompt: str, prompt_version: str, schema: type[BaseModel] | None = None) -> str:
        """Return the response text for `prompt`, from cache when possible.

        Only usable responses are cached: empty text is never cached, and when a
        schema is given the text must validate (otherwise ValidationError is raised
        and nothing is stored, so a rerun can recover).
        """
        key = self.cache_key(prompt, prompt_version, schema)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        text = self._call_with_backoff(prompt, schema, prompt_version)
        if schema is not None:
            schema.model_validate_json(text)  # raises ValidationError (empty text included)
        if not text.strip():
            return text
        self._cache[key] = text
        self._append_jsonl(self.cache_path, {
            "key": key, "model": self.model, "prompt_version": prompt_version, "response": text,
        })
        return text

    def generate_json(self, prompt: str, prompt_version: str, schema: type[BaseModel]) -> BaseModel:
        """Generate structured output and validate it with Pydantic (raises ValidationError)."""
        return schema.model_validate_json(self.generate(prompt, prompt_version, schema))

    def calls_today(self) -> int:
        """Real API calls logged for this model in the current Pacific quota day."""
        if not self.call_log_path.exists():
            return 0
        day = datetime.now(QUOTA_DAY_TZ).date()
        count = 0
        with open(self.call_log_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ts = datetime.fromisoformat(row["ts"]).astimezone(QUOTA_DAY_TZ)
                if row["model"] == self.model and ts.date() == day:
                    count += 1
        return count
