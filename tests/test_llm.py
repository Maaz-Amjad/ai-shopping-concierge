"""Tests for src/llm.py using a fake transport (never calls the real API)."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from src import llm
from src.llm import CallLimitReached, LLMClient, QuotaExhausted


class Echo(BaseModel):
    value: int


class FakeAPIError(Exception):
    def __init__(self, code: int, message: str = "error") -> None:
        super().__init__(message)
        self.code = code


class FakeTransport:
    """Returns queued outcomes: a string is a response, an exception is raised."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, model, prompt, schema, temperature, max_output_tokens) -> str:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    llm._last_call_at.clear()


def make_client(tmp_path, transport, **kw) -> LLMClient:
    return LLMClient(
        "fake-model", cache_path=tmp_path / "cache.jsonl", call_log_path=tmp_path / "calls.jsonl",
        rpm=6000, transport=transport, sleep=lambda s: None, **kw,
    )


def test_cache_hit_avoids_second_call_and_persists(tmp_path):
    transport = FakeTransport(['{"value": 1}'])
    client = make_client(tmp_path, transport)
    assert client.generate_json("p", "v1", Echo).value == 1
    assert client.generate_json("p", "v1", Echo).value == 1
    assert transport.calls == 1 and client.cache_hits == 1
    # A fresh client reads the cache from disk.
    reloaded = make_client(tmp_path, FakeTransport([]))
    assert reloaded.generate("p", "v1", Echo) == '{"value": 1}'


def test_prompt_version_changes_cache_key(tmp_path):
    transport = FakeTransport(['{"value": 1}', '{"value": 2}'])
    client = make_client(tmp_path, transport)
    client.generate("p", "v1", Echo)
    assert client.generate_json("p", "v2", Echo).value == 2
    assert transport.calls == 2


def test_retries_on_429_then_succeeds(tmp_path):
    transport = FakeTransport([FakeAPIError(429), FakeAPIError(503), '{"value": 3}'])
    client = make_client(tmp_path, transport)
    assert client.generate_json("p", "v1", Echo).value == 3
    assert client.calls_made == 3
    statuses = [json.loads(l)["status"] for l in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert statuses == ["error 429", "error 503", "ok"]


def test_persistent_429_raises_quota_exhausted(tmp_path):
    transport = FakeTransport([FakeAPIError(429)] * 3)
    client = make_client(tmp_path, transport, max_attempts=3)
    with pytest.raises(QuotaExhausted):
        client.generate("p", "v1")
    assert transport.calls == 3


def test_daily_quota_stops_immediately(tmp_path):
    transport = FakeTransport([FakeAPIError(429, "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel")])
    client = make_client(tmp_path, transport)
    with pytest.raises(QuotaExhausted):
        client.generate("p", "v1")
    assert transport.calls == 1


def test_dropped_connection_is_retried(tmp_path):
    # 27 Oct: "Server disconnected without sending a response" crashed the judge run.
    dropped = httpx.RemoteProtocolError("Server disconnected without sending a response.")
    client = make_client(tmp_path, FakeTransport([dropped, '{"value": 4}']))
    assert client.generate_json("p", "v1", Echo).value == 4
    assert client.calls_made == 2


def test_gemini_transport_sets_a_request_timeout(monkeypatch):
    # 27 Oct: a judge request hung for 8 hours because no timeout was set.
    from google import genai
    seen = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            return type("R", (), {"text": "ok"})()

    class FakeGenaiClient:
        def __init__(self, **kw):
            seen.update(kw)
            self.models = FakeModels()

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "Client", FakeGenaiClient)
    assert llm.gemini_transport("m", "p", None, 0.0, 10) == "ok"
    timeout = seen["http_options"].timeout
    assert timeout is not None and 0 < timeout <= 15 * 60 * 1000          # milliseconds


def test_non_retryable_error_is_raised(tmp_path):
    client = make_client(tmp_path, FakeTransport([FakeAPIError(400)]))
    with pytest.raises(FakeAPIError):
        client.generate("p", "v1")


def test_max_calls_limit(tmp_path):
    transport = FakeTransport(['{"value": 1}', '{"value": 2}'])
    client = make_client(tmp_path, transport, max_calls=1)
    client.generate("a", "v1")
    with pytest.raises(CallLimitReached):
        client.generate("b", "v1")
    assert transport.calls == 1


def test_invalid_json_raises_validation_error(tmp_path):
    client = make_client(tmp_path, FakeTransport(['{"value": "not a number"}']))
    with pytest.raises(ValidationError):
        client.generate_json("p", "v1", Echo)


def test_rate_limiter_is_shared_across_instances(tmp_path):
    slept: list[float] = []
    kw = dict(call_log_path=tmp_path / "l.jsonl", rpm=60, sleep=slept.append, clock=lambda: 100.0)
    a = LLMClient("fake-model", cache_path=tmp_path / "a.jsonl", transport=FakeTransport(["x"]), **kw)
    b = LLMClient("fake-model", cache_path=tmp_path / "b.jsonl", transport=FakeTransport(["y"]), **kw)
    a.generate("p", "v1")
    b.generate("q", "v1")  # different instance, same model -> must wait the full interval
    assert slept == [1.0]


def test_invalid_or_empty_response_is_not_cached(tmp_path):
    transport = FakeTransport(['{"value": "bad"}', "", '{"value": 5}'])
    client = make_client(tmp_path, transport)
    with pytest.raises(ValidationError):
        client.generate_json("p", "v1", Echo)
    with pytest.raises(ValidationError):
        client.generate_json("p", "v1", Echo)
    assert client.generate_json("p", "v1", Echo).value == 5  # not served from a bad cache entry
    assert transport.calls == 3


def test_cache_key_covers_model_schema_fields_and_settings(tmp_path):
    class EchoV2(BaseModel):  # same purpose, different fields
        value: int
        note: str = ""

    EchoV2.__name__ = "Echo"
    base = make_client(tmp_path, FakeTransport([]))
    key = base.cache_key("p", "v1", Echo)
    assert key != base.cache_key("p", "v1", EchoV2)
    assert key != make_client(tmp_path, FakeTransport([]), max_output_tokens=999).cache_key("p", "v1", Echo)
    other = LLMClient("other-model", cache_path=tmp_path / "c.jsonl", call_log_path=tmp_path / "l.jsonl")
    assert key != other.cache_key("p", "v1", Echo)


def test_calls_today_counts_logged_calls(tmp_path):
    client = make_client(tmp_path, FakeTransport(["a", "b"]))
    client.generate("a", "v1")
    client.generate("b", "v1")
    assert client.calls_today() == 2


def test_concurrent_calls_respect_the_call_cap_and_keep_the_cache_valid(tmp_path):
    # Verifier, Phase 11 (29 Oct): the API serves requests from a thread pool and shares one client per store.
    import threading
    import time

    llm._last_call_at.clear()
    client = LLMClient("m", tmp_path / "c.jsonl", tmp_path / "l.jsonl", rpm=1200, max_calls=2,
                       transport=lambda *a: (time.sleep(0.02), "ok")[1], sleep=time.sleep)
    results = []

    def worker(i):
        try:
            results.append(client.generate(f"prompt {i}", "v1"))
        except CallLimitReached:
            results.append("capped")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert client.calls_made == 2 and results.count("ok") == 2 and results.count("capped") == 6
    lines = (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(json.loads(line)["response"] == "ok" for line in lines)


TOOLS = [{"name": "search", "description": "find products", "parameters": {"type": "object", "properties": {
    "query": {"type": "string"}}, "required": ["query"]}}]


def test_tool_calls_are_cached_and_validated(tmp_path):
    llm._last_call_at.clear()
    calls = []

    def tool_transport(model, prompt, tools, temperature, max_tokens):
        calls.append(tools)
        return json.dumps({"name": "search", "args": {"query": "kettle"}}) if "kettle" in prompt else ""

    client = LLMClient("m", tmp_path / "c.jsonl", tmp_path / "l.jsonl", rpm=1e9, tool_transport=tool_transport,
                       transport=lambda *a: "plain", sleep=lambda s: None)
    first = client.generate_tool_call("find a kettle", "agent-v1", TOOLS)
    again = client.generate_tool_call("find a kettle", "agent-v1", TOOLS)
    assert first == again == {"name": "search", "args": {"query": "kettle"}} and len(calls) == 1
    assert client.generate("find a kettle", "agent-v1") == "plain"           # same prompt without tools: own key
    with pytest.raises(ValueError, match="no valid function call"):
        client.generate_tool_call("hello", "agent-v1", TOOLS)                 # empty response, not cached
    assert len((tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_cache_key_separates_providers_and_servers_but_keeps_gemini_keys(tmp_path):
    import hashlib

    def key(**kw):
        return LLMClient("m", tmp_path / "c.jsonl", tmp_path / "l.jsonl", transport=lambda *a: "", **kw).cache_key(
            "p", "v", None)

    old = hashlib.sha256(json.dumps(["m", "v", None, 0.0, 400, "p"], ensure_ascii=False,
                                    sort_keys=True).encode("utf-8")).hexdigest()
    assert key() == old                                               # existing Gemini cache stays valid
    keys = {key(), key(provider="openai"), key(provider="openai", base_url="http://localhost:11434/v1"),
            key(provider="openai", base_url="https://api.groq.com/openai/v1"), key(provider="anthropic")}
    assert len(keys) == 5


def test_invalid_tool_call_responses_are_not_cached(tmp_path):
    tools = [{"name": "search", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    replies = iter(['{"name": "bogus", "args": {}}', '{"name": "search", "args": {}}'])
    c = LLMClient("m", tmp_path / "c.jsonl", tmp_path / "l.jsonl", rpm=1e9, transport=lambda *a: "",
                  tool_transport=lambda *a: next(replies), sleep=lambda s: None)
    with pytest.raises(ValueError):
        c.generate_tool_call("p", "v", tools)
    assert c.generate_tool_call("p", "v", tools)["name"] == "search" and c.calls_made == 2
