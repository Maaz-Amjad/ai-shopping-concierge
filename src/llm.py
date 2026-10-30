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
import threading
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
_call_lock = threading.Lock()  # short critical sections only: call cap, rate-limit slot, cache append


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


def gemini_tool_transport(model: str, prompt: str, tools: list[dict], temperature: float,
                          max_output_tokens: int) -> str:
    """Call Gemini with function declarations and force exactly one function call (mode ANY, never executed by the
    SDK). Returns JSON text {"name": ..., "args": {...}}; the caller validates and runs it."""
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                              parameters_json_schema=t["parameters"]) for t in tools]
    config = types.GenerateContentConfig(
        temperature=temperature, max_output_tokens=max_output_tokens,
        tools=[types.Tool(function_declarations=declarations)],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    calls = response.function_calls or []
    if not calls:
        return ""                                   # not cached; the caller falls back
    return json.dumps({"name": calls[0].name, "args": dict(calls[0].args or {})}, ensure_ascii=False)


# ---- other providers: OpenAI-compatible (ChatGPT and open-weight servers) and Anthropic (Claude) ----------------
# Same contract as the Gemini transports: plain/structured text, or one forced tool call as {"name", "args"} JSON.
# The SDK's own retries are off (max_retries=0): LLMClient retries, caps and logs every real call itself.

def _api_key(env: str | None, required: bool = True) -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.environ.get(env or "", "")
    if required and not key:
        raise RuntimeError(f"set {env} in .env to use this provider")
    return key or "not-needed"                      # local open-weight servers usually need no key


def openai_transports(base_url: str | None = None, api_key_env: str | None = "OPENAI_API_KEY"):
    """ChatGPT, or any OpenAI-compatible server for open-weight models (Ollama, vLLM, LM Studio, Groq, Together...)
    when `base_url` is set, e.g. http://localhost:11434/v1 for Ollama."""
    def client():
        from openai import OpenAI

        return OpenAI(api_key=_api_key(api_key_env, required=base_url is None), base_url=base_url,
                      timeout=REQUEST_TIMEOUT_MS / 1000, max_retries=0)

    def transport(model: str, prompt: str, schema: type[BaseModel] | None, temperature: float,
                  max_output_tokens: int) -> str:
        extra = {"response_format": {"type": "json_schema", "json_schema": {
            "name": schema.__name__, "schema": schema.model_json_schema()}}} if schema else {}
        r = client().chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                             temperature=temperature, max_tokens=max_output_tokens, seed=42, **extra)
        return r.choices[0].message.content or ""

    def tool_transport(model: str, prompt: str, tools: list[dict], temperature: float, max_output_tokens: int) -> str:
        r = client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature,
            max_tokens=max_output_tokens, seed=42, tool_choice="required",
            tools=[{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                     "parameters": t["parameters"]}} for t in tools])
        calls = r.choices[0].message.tool_calls or []
        if not calls:
            return ""
        return json.dumps({"name": calls[0].function.name, "args": json.loads(calls[0].function.arguments or "{}")},
                          ensure_ascii=False)

    return transport, tool_transport


def anthropic_transports(base_url: str | None = None, api_key_env: str | None = "ANTHROPIC_API_KEY"):
    """Claude. Structured output uses a forced tool whose input schema is the expected JSON. The current API takes no
    temperature, so repeatability on Claude comes from the response cache."""
    def client():
        from anthropic import Anthropic

        return Anthropic(api_key=_api_key(api_key_env), base_url=base_url, timeout=REQUEST_TIMEOUT_MS / 1000,
                         max_retries=0)

    def first_tool_use(response):
        return next((b for b in response.content if getattr(b, "type", "") == "tool_use"), None)

    def transport(model: str, prompt: str, schema: type[BaseModel] | None, temperature: float,
                  max_output_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        if schema is None:
            r = client().messages.create(model=model, max_tokens=max_output_tokens, messages=messages)
            return "".join(getattr(b, "text", "") for b in r.content)
        tool = {"name": "respond", "description": "Return the answer in this exact structure.",
                "input_schema": schema.model_json_schema()}
        r = client().messages.create(model=model, max_tokens=max_output_tokens, messages=messages, tools=[tool],
                                     tool_choice={"type": "tool", "name": "respond"})
        block = first_tool_use(r)
        return json.dumps(block.input, ensure_ascii=False) if block else ""

    def tool_transport(model: str, prompt: str, tools: list[dict], temperature: float, max_output_tokens: int) -> str:
        r = client().messages.create(
            model=model, max_tokens=max_output_tokens, messages=[{"role": "user", "content": prompt}],
            tool_choice={"type": "any"},
            tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                   for t in tools])
        block = first_tool_use(r)
        return json.dumps({"name": block.name, "args": dict(block.input or {})}, ensure_ascii=False) if block else ""

    return transport, tool_transport


PROVIDERS = ("gemini", "openai", "anthropic")


def transports_for(provider: str = "gemini", base_url: str | None = None, api_key_env: str | None = None):
    """(transport, tool_transport) for a provider named in config.yaml (llm.provider)."""
    if provider == "gemini":
        return gemini_transport, gemini_tool_transport
    if provider == "openai":
        return openai_transports(base_url, api_key_env or "OPENAI_API_KEY")
    if provider == "anthropic":
        return anthropic_transports(base_url, api_key_env or "ANTHROPIC_API_KEY")
    raise ValueError(f"unknown llm provider {provider!r}; use one of {PROVIDERS}")


def provider_settings(llm_cfg: dict) -> dict:
    """LLMClient keyword arguments for the provider part of config.yaml's llm section."""
    return {"provider": llm_cfg.get("provider", "gemini"), "base_url": llm_cfg.get("base_url"),
            "api_key_env": llm_cfg.get("api_key_env")}


def _error_code(exc: BaseException) -> int | None:
    """HTTP-like status code of an API error, if it has one."""
    for attr in ("code", "status_code"):                # Gemini errors use .code, OpenAI/Anthropic .status_code
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    return None


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
        tool_transport: Callable[..., str] | None = None,
        provider: str = "gemini",
        base_url: str | None = None,
        api_key_env: str | None = None,
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
        default, default_tools = transports_for(provider, base_url, api_key_env)
        self.provider = provider
        self.base_url = base_url
        self.transport = transport or default
        self.tool_transport = tool_transport or default_tools
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
                    try:
                        row = json.loads(line)
                        cache[row["key"]] = row["response"]
                    except (ValueError, KeyError, TypeError):
                        continue                  # a torn line from a crash mid-write is skipped
        return cache

    def cache_key(self, prompt: str, prompt_version: str, schema: type[BaseModel] | None) -> str:
        parts: list = [self.model, prompt_version, schema.model_json_schema() if schema else None,
                       self.temperature, self.max_output_tokens, prompt]
        if self.provider != "gemini" or self.base_url:     # same model name on two servers: separate entries
            parts.append({"provider": self.provider, "base_url": self.base_url})   # (Gemini keys unchanged)
        payload = json.dumps(parts, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _append_jsonl(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- calling -------------------------------------------------------------
    def _wait_for_rate_limit(self) -> None:
        """Reserve the next call slot under the lock, then wait outside it (other threads keep going)."""
        with _call_lock:
            now = self.clock()
            last = _last_call_at.get(self.model)
            slot = now if last is None else max(now, last + self.min_interval)
            _last_call_at[self.model] = slot
        if slot - now > 0:
            self.sleep(slot - now)

    def _call_with_backoff(self, prompt: str, schema: type[BaseModel] | None, prompt_version: str,
                           tools: list[dict] | None = None) -> str:
        for attempt in range(1, self.max_attempts + 1):
            with _call_lock:                      # the cap is shared by all threads using this client
                if self.max_calls is not None and self.calls_made >= self.max_calls:
                    raise CallLimitReached(f"max_calls={self.max_calls} reached for {self.model}")
                self.calls_made += 1
            self._wait_for_rate_limit()
            try:
                if tools:
                    text = self.tool_transport(self.model, prompt, tools, self.temperature, self.max_output_tokens)
                else:
                    text = self.transport(self.model, prompt, schema, self.temperature, self.max_output_tokens)
                self._log_call(prompt_version, "ok")
                return text
            except Exception as exc:  # noqa: BLE001 - classified below
                code = _error_code(exc)
                self._log_call(prompt_version, f"error {code}")
                if code == 429 and _is_daily_quota(exc):
                    raise QuotaExhausted(f"daily quota exhausted for {self.model}") from exc
                # Dropped connections and timeouts have no status code but are transient.
                transient = isinstance(exc, httpx.TransportError) or type(exc).__name__ in (
                    "APIConnectionError", "APITimeoutError")     # OpenAI/Anthropic SDK network errors
                if code not in RETRYABLE_CODES and not transient:
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
        with _call_lock:
            self._log_unlocked(prompt_version, status)

    def _log_unlocked(self, prompt_version: str, status: str) -> None:
        self._append_jsonl(self.call_log_path, {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": self.model, "prompt_version": prompt_version, "status": status,
        })

    def generate(self, prompt: str, prompt_version: str, schema: type[BaseModel] | None = None,
                 tools: list[dict] | None = None) -> str:
        """Return the response text for `prompt`, from cache when possible.

        Only usable responses are cached: empty text is never cached, and when a
        schema is given the text must validate (otherwise ValidationError is raised
        and nothing is stored, so a rerun can recover). With `tools`, the response is
        one forced function call as JSON (see gemini_tool_transport).
        """
        keyed = prompt + "\n#tools " + json.dumps(tools, sort_keys=True) if tools else prompt
        key = self.cache_key(keyed, prompt_version, schema)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        return self._call_and_store(key, prompt, schema, prompt_version, tools)    # calls run in parallel

    def _call_and_store(self, key: str, prompt: str, schema: type[BaseModel] | None, prompt_version: str,
                        tools: list[dict] | None = None) -> str:
        text = self._call_with_backoff(prompt, schema, prompt_version, tools)
        if tools:                             # cache only a call to one of the offered functions
            call = json.loads(text) if text.strip() else {}
            if not isinstance(call, dict) or call.get("name") not in {t["name"] for t in tools}:
                raise ValueError(f"no valid function call in response: {text[:200]!r}")
        if schema is not None:
            schema.model_validate_json(text)  # raises ValidationError (empty text included)
        if not text.strip():
            return text
        with _call_lock:
            if key not in self._cache:
                self._cache[key] = text
                self._append_jsonl(self.cache_path, {
                    "key": key, "model": self.model, "prompt_version": prompt_version, "response": text,
                })
        return text

    def generate_json(self, prompt: str, prompt_version: str, schema: type[BaseModel]) -> BaseModel:
        """Generate structured output and validate it with Pydantic (raises ValidationError)."""
        return schema.model_validate_json(self.generate(prompt, prompt_version, schema))

    def generate_tool_call(self, prompt: str, prompt_version: str, tools: list[dict]) -> dict:
        """One forced function call: {"name", "args"}; ValueError when there is none or the name is unknown."""
        text = self.generate(prompt, prompt_version, tools=tools)
        call = json.loads(text) if text.strip() else {}
        if call.get("name") not in {t["name"] for t in tools}:
            raise ValueError(f"no valid function call in response: {text[:200]!r}")
        call.setdefault("args", {})
        return call

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
