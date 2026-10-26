"""Tests for src/llm.py using a fake transport (never calls the real API)."""

from __future__ import annotations

import json

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
