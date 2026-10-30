"""Tests for src/parse.py with a fake LLM transport (never calls the real API)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import llm
from src.llm import CallLimitReached, LLMClient, QuotaExhausted
from src.parse import CatalogMeta, ParsedQuery, build_prompt, parse_query, validate

META = CatalogMeta(categories=["Ice Makers", "Range Hoods", "Other"],
                   brands=["FRIGIDAIRE", "GE", "Whirlpool"], prompt_brands=["FRIGIDAIRE", "GE", "Whirlpool"])


class FakeAPIError(Exception):
    def __init__(self, code: int, message: str = "error") -> None:
        super().__init__(message)
        self.code = code


class FakeTransport:
    def __init__(self, outcomes: list) -> None:
        self.outcomes, self.prompts = list(outcomes), []

    def __call__(self, model, prompt, schema, temperature, max_output_tokens):
        self.prompts.append(prompt)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, str) else json.dumps(outcome)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    llm._last_call_at.clear()


def client(tmp_path, outcomes, **kw) -> tuple[LLMClient, FakeTransport]:
    transport = FakeTransport(outcomes)
    return LLMClient("fake", cache_path=tmp_path / "c.jsonl", call_log_path=tmp_path / "l.jsonl", rpm=6000,
                     transport=transport, sleep=lambda s: None, max_attempts=2, **kw), transport


def pq(**kw) -> dict:
    base = {"semantic_query": "ice maker", "max_price": None, "min_price": None,
            "category": None, "brand": None, "exclude_terms": []}
    return {**base, **kw}


def test_catalog_meta_uses_most_frequent_brand_spelling():
    df = pd.DataFrame({"brand": ["FRIGIDAIRE"] * 3 + ["Frigidaire", "GE", None],
                       "category": ["Ice Makers"] * 6})
    meta = CatalogMeta.from_catalog(df)
    assert meta.brands == ["FRIGIDAIRE", "GE"] and meta.categories == ["Ice Makers"]


def test_prompt_lists_catalog_values_and_rules_and_ends_with_query():
    prompt = build_prompt("quiet ice maker under $150", META)
    assert '"Ice Makers"' in prompt and '"Whirlpool"' in prompt
    assert "compatible with" in prompt and "not instructions" in prompt
    assert prompt.rstrip().endswith("Query: quiet ice maker under $150\nJSON:")
    assert "previous answer was invalid" in build_prompt("q", META, error="bad json")


def test_valid_parse(tmp_path):
    c, _ = client(tmp_path, [pq(max_price=150, category="Ice Makers")])
    result = parse_query("quiet ice maker under $150", META, c)
    assert not result.fallback and result.llm_attempts == 1 and result.log == []
    assert result.parsed.max_price == 150 and result.parsed.category == "Ice Makers"


@pytest.mark.parametrize("raw, expected, log_part", [
    (pq(category="ice makers"), {"category": "Ice Makers"}, None),            # case-insensitive
    (pq(category="Ice Maker"), {"category": "Ice Makers"}, "fuzzy-matched"),  # near miss
    (pq(category="Toasters"), {"category": None}, "not in catalog"),          # invented -> dropped
    (pq(brand="frigidaire"), {"brand": "FRIGIDAIRE"}, None),
    (pq(brand="Samsung"), {"brand": None}, "not in catalog"),
    (pq(min_price=50, max_price=20), {"min_price": None, "max_price": None}, "both dropped"),
    (pq(max_price=-5), {"max_price": None}, "negative"),
    (pq(semantic_query="  "), {"semantic_query": "raw text"}, "raw query used"),
    (pq(exclude_terms=["Stainless", "stainless ", ""]), {"exclude_terms": ["stainless"]}, None),
])
def test_validate(raw, expected, log_part):
    parsed, log = validate(ParsedQuery(**raw), "raw text", META)
    for key, value in expected.items():
        assert getattr(parsed, key) == value
    assert (log_part is None and log == []) or any(log_part in line for line in log)


def test_near_miss_never_maps_to_a_different_real_value():
    meta = CatalogMeta(
        categories=["Oven Parts & Accessories", "Washer Parts & Accessories", "Dishwasher Parts & Accessories",
                    "Range Parts & Accessories", "Range Hood Parts & Accessories", "Ice Makers"],
        brands=["Capresso", "Appliance Pros"], prompt_brands=[])
    for wrong in ["Microwave Parts & Accessories", "Dishwasher Parts", "Range Hood Parts"]:
        parsed, log = validate(ParsedQuery(semantic_query="x", category=wrong), "x", meta)
        assert parsed.category is None and "not in catalog" in log[0], wrong
    parsed, _ = validate(ParsedQuery(semantic_query="x", brand="Capmesso"), "x", meta)
    assert parsed.brand is None
    # Harmless spelling variants still resolve.
    for variant, expected in [("Ice Maker", "Ice Makers"), ("oven parts and accessories", "Oven Parts & Accessories"),
                              ("appliance-pros", "Appliance Pros")]:
        field = "brand" if expected == "Appliance Pros" else "category"
        parsed, _ = validate(ParsedQuery(semantic_query="x", **{field: variant}), "x", meta)
        assert getattr(parsed, field) == expected, variant


def test_retry_once_with_error_then_success(tmp_path):
    c, transport = client(tmp_path, ["not json", pq(brand="GE")])
    result = parse_query("GE filter", META, c)
    assert not result.fallback and result.llm_attempts == 2 and result.parsed.brand == "GE"
    assert "previous answer was invalid" in transport.prompts[1]


def test_two_invalid_outputs_fall_back_to_semantic_only(tmp_path):
    c, _ = client(tmp_path, ["not json", '{"semantic_query": 5'])
    result = parse_query("GE filter under $20", META, c)
    assert result.fallback and result.parsed == ParsedQuery(semantic_query="GE filter under $20")
    assert "parser fallback" in result.log[0]


def test_api_error_falls_back(tmp_path):
    c, _ = client(tmp_path, [FakeAPIError(400)])
    assert parse_query("ice maker", META, c).fallback


def test_quota_exhausted_raises_by_default_and_can_fall_back(tmp_path):
    c, _ = client(tmp_path, [FakeAPIError(429, "GenerateRequestsPerDay")])
    with pytest.raises(QuotaExhausted):
        parse_query("ice maker", META, c)
    c2, _ = client(tmp_path, [FakeAPIError(429, "GenerateRequestsPerDay")])
    assert parse_query("ice maker", META, c2, raise_on_quota=False).fallback


def test_call_limit_stops_instead_of_falling_back(tmp_path):
    c, _ = client(tmp_path, [], max_calls=0)
    with pytest.raises(CallLimitReached):
        parse_query("ice maker", META, c)


def test_cached_parse_needs_no_second_call(tmp_path):
    c, transport = client(tmp_path, [pq()])
    parse_query("ice maker", META, c)
    parse_query("ice maker", META, c)
    assert len(transport.prompts) == 1


def test_empty_query_rejected(tmp_path):
    c, _ = client(tmp_path, [])
    with pytest.raises(ValueError):
        parse_query("   ", META, c)


# ---- Phase 13.7 (G): a category is kept only when the request states it -----------------------------------

def test_category_stated_accepts_name_or_singular_only():
    from src.parse import category_stated

    assert category_stated("Ice Makers", "quiet portable countertop ice maker under $150")
    assert category_stated("Range Hoods", "RANGE HOODS for small kitchens")
    assert category_stated("Beverage Refrigerators", "beverage refrigerator for cans")
    assert category_stated("Espresso Machine Accessories", "ok, show me espresso machine accessories")
    assert category_stated("Espresso Machine Accessories", "an espresso machine accessory")
    assert not category_stated("Ice Makers", "make chewable nugget ice at home for parties")
    assert not category_stated("Range Hoods", "hood ornament")


def test_parse_query_drops_a_category_the_shopper_did_not_state(tmp_path):
    c, _ = client(tmp_path, [pq(semantic_query="nugget ice at home", category="Ice Makers"),
                             pq(semantic_query="ice maker", category="Ice Makers")])
    out = parse_query("make chewable nugget ice at home", META, c)
    assert out.parsed.category is None and any("not stated" in line for line in out.log)
    assert parse_query("an ice maker for parties", META, c).parsed.category == "Ice Makers"
    kept, _ = validate(ParsedQuery(semantic_query="cold drinks", category="Ice Makers"), "cold drinks", META)
    assert kept.category == "Ice Makers"                     # validate alone (used for carried-over limits) keeps it
