"""Provider adapters (OpenAI-compatible, Anthropic) with fake SDK clients: no keys, no network."""

import json
from types import SimpleNamespace as NS

import openai
import anthropic
import pytest
from pydantic import BaseModel

from src import llm
from src.llm import LLMClient, transports_for

TOOLS = [{"name": "search", "description": "find", "parameters": {"type": "object", "properties": {
    "request": {"type": "string"}}, "required": ["request"]}}]


class Reply(BaseModel):
    summary: str


class FakeOpenAI:
    calls: list = []

    def __init__(self, **kwargs):
        FakeOpenAI.calls.append(("init", kwargs))
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        FakeOpenAI.calls.append(("create", kwargs))
        if "tools" in kwargs:
            call = NS(function=NS(name="search", arguments=json.dumps({"request": "kettle"})))
            return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))])
        return NS(choices=[NS(message=NS(content='{"summary": "ok"}', tool_calls=None))])


class FakeAnthropic:
    calls: list = []

    def __init__(self, **kwargs):
        FakeAnthropic.calls.append(("init", kwargs))
        self.messages = NS(create=self.create)

    def create(self, **kwargs):
        FakeAnthropic.calls.append(("create", kwargs))
        if "tools" not in kwargs:
            return NS(content=[NS(type="text", text="plain answer")])
        name = kwargs["tools"][0]["name"]
        data = {"summary": "ok"} if name == "respond" else {"request": "kettle"}
        return NS(content=[NS(type="text", text="thinking"), NS(type="tool_use", name=name, input=data)])


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    llm._last_call_at.clear()
    FakeOpenAI.calls, FakeAnthropic.calls = [], []
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def client(tmp_path, provider, **kw):
    return LLMClient("some-model", tmp_path / "c.jsonl", tmp_path / "l.jsonl", rpm=1e9, provider=provider,
                     sleep=lambda s: None, **kw)


def test_openai_structured_output_and_tool_calls(tmp_path):
    c = client(tmp_path, "openai")
    assert c.generate_json("summarize", "v1", Reply).summary == "ok"
    create = [k for kind, k in FakeOpenAI.calls if kind == "create"][0]
    assert create["response_format"]["json_schema"]["name"] == "Reply" and create["temperature"] == 0.0
    assert c.generate_tool_call("find a kettle", "agent-v1", TOOLS) == {"name": "search", "args": {"request": "kettle"}}
    tool_create = [k for kind, k in FakeOpenAI.calls if kind == "create"][1]
    assert tool_create["tool_choice"] == "required" and tool_create["tools"][0]["function"]["name"] == "search"
    assert all(k.get("max_retries") == 0 for kind, k in FakeOpenAI.calls if kind == "init")   # our retries only


def test_open_weight_server_needs_only_a_base_url(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    c = client(tmp_path, "openai", base_url="http://localhost:11434/v1")
    assert c.generate("hi", "v1") == '{"summary": "ok"}'
    init = [k for kind, k in FakeOpenAI.calls if kind == "init"][0]
    assert init["base_url"] == "http://localhost:11434/v1" and init["api_key"] == "not-needed"
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        client(tmp_path, "openai").generate("hello again", "v1")       # ChatGPT itself needs a key


def test_anthropic_structured_output_plain_text_and_tool_calls(tmp_path):
    c = client(tmp_path, "anthropic")
    assert c.generate_json("summarize", "v1", Reply).summary == "ok"
    first = [k for kind, k in FakeAnthropic.calls if kind == "create"][0]
    assert first["tool_choice"] == {"type": "tool", "name": "respond"} and "temperature" not in first
    assert c.generate("say hi", "v1") == "plain answer"
    assert c.generate_tool_call("find a kettle", "agent-v1", TOOLS) == {"name": "search", "args": {"request": "kettle"}}
    last = [k for kind, k in FakeAnthropic.calls if kind == "create"][-1]
    assert last["tool_choice"] == {"type": "any"} and last["tools"][0]["input_schema"]["required"] == ["request"]


def test_unknown_provider_and_sdk_errors_are_classified():
    with pytest.raises(ValueError, match="unknown llm provider"):
        transports_for("mystery")
    assert llm._error_code(NS(status_code=429)) == 429 and llm._error_code(NS(code=503)) == 503


def test_sdk_connection_errors_are_retried(tmp_path):
    class APIConnectionError(Exception):
        pass

    outcomes = iter([APIConnectionError("reset"), "fine"])

    def transport(*a):
        o = next(outcomes)
        if isinstance(o, Exception):
            raise o
        return o

    c = LLMClient("m", tmp_path / "c.jsonl", tmp_path / "l.jsonl", rpm=1e9, transport=transport,
                  sleep=lambda s: None, provider="openai")
    assert c.generate("q", "v1") == "fine" and c.calls_made == 2
