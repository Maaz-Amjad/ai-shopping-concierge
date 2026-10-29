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


def test_search_products_explicit_filters_override_and_invalid_ones_dropped(tmp_path):
    from src.pipeline import search_products

    parsed = json.dumps({"semantic_query": "ice maker", "max_price": 500})
    res = resources(tmp_path, [parsed, gen(["A"], "One GE ice maker.")])
    out = search_products("ice maker under $500", res, max_price=110, brand="Samsung")
    assert out.parsed_query.max_price == 110 and out.parsed_query.brand is None   # explicit wins; unknown brand dropped
    assert out.ranked["product_id"].tolist() == ["A"]
    assert "explicit filter: max_price=110" in out.flags["parse_log"]
    assert any("brand" in line and "Samsung" in line for line in out.flags["parse_log"])


def test_get_similar_excludes_itself_and_rejects_unknown_ids(tmp_path):
    from src.pipeline import get_similar

    res = resources(tmp_path, [])
    sim = get_similar("A", res, n=2)
    assert sim["product_id"].tolist() == ["B", "C"] and sim["similarity"].tolist() == [0.8, 0.0]
    with pytest.raises(KeyError):
        get_similar("nope", res)


def test_result_to_dict_is_json_ready_with_scores_and_reasons(tmp_path):
    from src.pipeline import result_to_dict

    res = resources(tmp_path, [gen(["A", "B"])])
    out = respond("ice maker", res, parsed=ParsedQuery(semantic_query="ice maker", category="Ice Makers"))
    d = json.loads(json.dumps(result_to_dict("ice maker", out, res)))      # round-trips through JSON
    assert [p["product_id"] for p in d["products"]] == ["A", "B"]
    assert d["products"][0]["reason"] == "fits" and {"s_sem", "s_pers", "s_pop", "final"} <= set(d["products"][0])
    assert d["parsed"]["category"] == "Ice Makers" and d["flags"]["cold_start"] is True


def test_search_products_applies_not_x_exclusions(tmp_path):
    # 29 Oct: the app shows "not ..." as an understood limit, so the tool path must enforce it (spec 8.3, S5x).
    from src.pipeline import search_products

    parsed = json.dumps({"semantic_query": "ice maker", "exclude_terms": ["maker b"]})
    res = resources(tmp_path, [parsed, gen(["A", "C"])])
    out = search_products("ice maker, not maker b", res)
    assert "B" not in out.ranked["product_id"].tolist()


# ---- Phase 13.1: ratings in the product views and production weights ------------------------------------

def test_app_config_uses_production_weights_and_results_show_ratings(tmp_path):
    from src.pipeline import app_config, result_to_dict

    cfg = {"ranking": {"w_sem": 0.7, "w_pers": 0.25, "w_pop": 0.05},
           "app": {"ranking": {"w_sem": 0.6, "w_pers": 0.25, "w_pop": 0.05, "w_rat": 0.1}}}
    assert app_config(cfg)["ranking"]["w_rat"] == 0.1 and cfg["ranking"] == {"w_sem": 0.7, "w_pers": 0.25,
                                                                               "w_pop": 0.05}
    assert app_config({"ranking": {"w_sem": 1}})["ranking"] == {"w_sem": 1}          # no app section: unchanged
    res = resources(tmp_path, [gen(["A", "B"])])
    res.catalog = res.catalog.assign(avg_rating=[4.5, np.nan, 3.0], rating_count=[120, 0, 4])
    out = respond("ice maker", res, parsed=ParsedQuery(semantic_query="ice maker", category="Ice Makers"))
    products = {p["product_id"]: p for p in result_to_dict("ice maker", out, res)["products"]}
    assert products["A"]["avg_rating"] == 4.5 and products["A"]["rating_count"] == 120
    assert products["B"]["avg_rating"] is None and "s_rat" in products["A"]



def test_similar_products_skip_out_of_stock_items(tmp_path):
    from src.pipeline import get_similar

    res = resources(tmp_path, [])
    res.catalog = res.catalog.assign(in_stock=[True, False, None])
    assert get_similar("A", res, 2)["product_id"].tolist() == ["C"]



def test_no_result_answer_says_when_matches_are_out_of_stock(tmp_path):
    res = resources(tmp_path, [])
    res.catalog = res.catalog.assign(in_stock=[False, False, True])
    out = respond("ice maker", res, parsed=ParsedQuery(semantic_query="ice maker", category="Ice Makers"))
    assert out.ranked.empty and "2 matching products are out of stock" in out.answer.text


def test_personal_weight_shrinks_only_for_returning_shoppers_with_stated_limits(tmp_path):
    cfg = {**CFG, "ranking": {**CFG["ranking"], "pers_factor_with_limits": 0.0}}
    res = resources(tmp_path, [gen(["A", "B"])] * 3)
    res.cfg = cfg
    limited = ParsedQuery(semantic_query="ice maker", category="Ice Makers")
    warm = respond("ice maker", res, user_id="u_lg", parsed=limited).ranked.set_index("product_id")
    assert warm.loc["B", "final"] == pytest.approx(0.95 * 0 + 0.05 * warm.loc["B", "s_pop"])   # personal share gone
    open_query = ParsedQuery(semantic_query="ice maker")
    free = respond("ice maker", res, user_id="u_lg", parsed=open_query).ranked.set_index("product_id")
    assert free.loc["B", "final"] == pytest.approx(0.25 * free.loc["B", "s_pers"] + 0.7 * free.loc["B", "s_sem"]
                                                   + 0.05 * free.loc["B", "s_pop"])     # no limits: base weights


def test_respond_uses_hybrid_retrieval_when_configured(tmp_path):
    from src.pipeline import app_config
    from src.retrieve import BM25Retriever

    res = resources(tmp_path, [gen(["B", "A"])])
    res.bm25 = BM25Retriever(["ice maker a", "ice maker b steel model x200", "hood c"])
    res.cfg = {**CFG, "retrieval": {**CFG["retrieval"], "hybrid_alpha": 1.0}}
    out = respond("x200 ice maker", res, parsed=ParsedQuery(semantic_query="x200 ice maker", category="Ice Makers"))
    assert out.ranked["product_id"].tolist()[0] == "B"             # keyword match first (dense prefers A)
    cfg = {"ranking": {}, "retrieval": {"pool_size": 5}, "app": {"retrieval": {"hybrid_alpha": 0.2}}}
    assert app_config(cfg)["retrieval"] == {"pool_size": 5, "hybrid_alpha": 0.2}


# ---- Phase 13.6 (C): products first, answer when validated -------------------------------------------------

def test_products_are_handed_over_before_the_answer_is_generated(tmp_path):
    events = []
    res = resources(tmp_path, [])
    answers = iter([gen(["A", "B"])])
    res.client.transport = lambda *a: events.append("generate") or next(answers)
    out = respond("ice maker", res, parsed=ParsedQuery(semantic_query="ice maker", category="Ice Makers"),
                  on_products=lambda view: events.append(("products", [p["product_id"] for p in view["products"]])))
    assert events == [("products", ["A", "B"]), "generate"] and out.answer.answer is not None
