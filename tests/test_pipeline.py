"""End-to-end test of src/pipeline.respond on a toy catalog with a fake encoder and fake LLM."""

import json

import numpy as np
import pandas as pd
import pytest

from src import llm
from src.llm import LLMClient
from src.parse import CatalogMeta, ParsedQuery
from src.pipeline import Resources, respond
from src.retrieve import DenseIndex

CFG = {"seed": 42, "embedding": {"query_instruction": False},
       "retrieval": {"pool_size": 10, "k0": 2, "k_max": 8},
       "ranking": {"w_sem": 0.7, "w_pers": 0.25, "w_pop": 0.05}, "generation": {"top_n": 2}}


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    llm._last_call_at.clear()


class FakeModel:
    """Every query encodes to direction [1, 0]."""

    def encode(self, texts, prompt=None, normalize_embeddings=True, convert_to_numpy=True):
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


def resources(tmp_path, responses: list[str]) -> Resources:
    catalog = pd.DataFrame({"product_id": ["A", "B", "C"], "title": ["ice maker a", "ice maker b", "hood c"],
                            "brand": ["GE", "LG", "GE"], "category": ["Ice Makers", "Ice Makers", "Range Hoods"],
                            "price": [100.0, 120.0, np.nan], "features": ["", "", ""],
                            "description": ["", "", ""], "popularity": [50, 50, 50]})
    e = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
    it = iter(responses)
    client = LLMClient("m", tmp_path / "c.jsonl", tmp_path / "l.jsonl", rpm=1e9, transport=lambda *a: next(it),
                       sleep=lambda s: None)
    return Resources(CFG, catalog, e, DenseIndex(e), FakeModel(), CatalogMeta.from_catalog(catalog),
                     {"u_lg": np.array([0.8, 0.6], dtype=np.float32)}, client, ["GE", "LG"])


def gen(ids: list[str], summary="Two ice makers.") -> str:
    return json.dumps({"summary": summary, "recommendations": [{"product_id": i, "reason": "fits"} for i in ids]})


def test_respond_cold_start_vs_personalized_and_grounded(tmp_path):
    pq = ParsedQuery(semantic_query="ice maker", category="Ice Makers")
    res = resources(tmp_path, [gen(["A", "B"]), gen(["B", "A"])])
    cold = respond("ice maker", res, parsed=pq)
    assert cold.ranked["product_id"].tolist() == ["A", "B"] and cold.flags["cold_start"]
    assert cold.answer.fallback is None and cold.answer.errors == []
    assert set(cold.timings_ms) == {"parse", "retrieve", "rank", "generate", "total"}
    warm = respond("ice maker", res, user_id="u_lg", parsed=pq)
    # B: s_sem 0, s_pers 1 -> 0.25 + 0.05; A: s_sem 1, s_pers 0 -> 0.7 + 0.05. A still first, components shown.
    assert warm.ranked["product_id"].tolist() == ["A", "B"] and not warm.flags["cold_start"]
    assert warm.ranked.set_index("product_id").loc["B", "s_pers"] == 1.0


def test_respond_parses_with_llm_and_reports_no_result(tmp_path):
    parsed = json.dumps({"semantic_query": "range hood", "max_price": 50, "category": "Range Hoods"})
    res = resources(tmp_path, [parsed])
    out = respond("range hood under $50", res)
    assert out.parsed_query.max_price == 50 and out.ranked.empty
    assert out.flags["price_excluded"] == 1 and out.answer.fallback == "no_result"
    assert "Removing the price constraint would give 1 matching products." in out.answer.text


def test_five_queries_end_to_end(tmp_path):
    """Spec 13 integration test: five queries, cached/fake LLM, every path returns a usable answer."""
    pqs = [ParsedQuery(semantic_query="ice maker"),                                     # no filters
           ParsedQuery(semantic_query="ice maker", brand="LG"),                         # brand filter
           ParsedQuery(semantic_query="ice maker", max_price=110),                      # price filter
           ParsedQuery(semantic_query="hood", category="Range Hoods", max_price=50),    # unpriced only
           None]                                                                        # parser fallback
    res = resources(tmp_path, [gen(["A", "B"]), gen(["B"], "One LG ice maker."), gen(["A"], "One under $110."),
                               "not json", "still not json", gen(["A", "B"])])
    outs = [respond("q", res, parsed=pq) for pq in pqs]
    assert [o.ranked["product_id"].tolist() for o in outs] == [["A", "B"], ["B"], ["A"], [], ["A", "B"]]
    assert [o.answer.fallback for o in outs] == [None, None, None, "no_result", None]
    assert "Price not listed (cannot confirm they fit your budget):\n- hood c by GE" in outs[3].answer.text
    assert outs[4].flags["parser_fallback"] and outs[4].parsed_query.semantic_query == "q"
    assert all(o.answer.text for o in outs)
