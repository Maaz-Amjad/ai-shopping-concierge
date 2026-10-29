"""Tests for src/retrieve.py on small synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.parse import ParsedQuery
from src.retrieve import BM25Retriever, DenseIndex, TfidfRetriever, retrieve, tokenize, top_k

DOCS = [
    "Whirlpool dryer belt W10198086 replacement",
    "Samsung refrigerator water filter DA29-00020B",
    "Portable countertop ice maker, quiet, 26 lbs per day",
    "GE dishwasher rack wheel kit",
]


def unit_rows(n: int, dim: int = 8, seed: int = 42) -> np.ndarray:
    m = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_tokenize_keeps_part_numbers():
    assert tokenize("Filter DA29-00020B, 4.6 Cu ft!") == ["filter", "da29-00020b", "4.6", "cu", "ft"]


def test_top_k_breaks_ties_by_row():
    out = top_k(np.array([0.5, 0.9, 0.5, 0.9]), k=4)
    assert out["row"].tolist() == [1, 3, 0, 2] and out["rank"].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("retriever_cls", [TfidfRetriever, BM25Retriever])
def test_lexical_retrievers_find_exact_part_number(retriever_cls):
    res = retriever_cls(DOCS).search("da29-00020b filter", k=2)
    assert res["row"].iloc[0] == 1 and len(res) == 2


def test_tfidf_save_load(tmp_path):
    r = TfidfRetriever(DOCS)
    r.save(tmp_path / "tfidf.pkl")
    again = TfidfRetriever.load(tmp_path / "tfidf.pkl")
    assert again.search("ice maker", 1)["row"].item() == r.search("ice maker", 1)["row"].item() == 2


def test_dense_search_matches_brute_force():
    emb = unit_rows(50)
    q = emb[7] * 0.9 + emb[3] * 0.1
    res = DenseIndex(emb).search(q, k=5)
    expected = np.argsort(-(emb @ q), kind="stable")[:5]
    assert res["row"].tolist() == expected.tolist()
    assert np.allclose(res["score"], (emb @ q)[expected], atol=1e-5)


def test_filter_first_only_returns_allowed_rows_and_is_exact():
    emb = unit_rows(50)
    q = emb[0]
    mask = np.zeros(50, dtype=bool)
    mask[[2, 9, 17, 30, 44]] = True
    res = DenseIndex(emb).filter_first(q, mask, k=3)
    assert set(res["row"]) <= {2, 9, 17, 30, 44} and len(res) == 3
    allowed = np.flatnonzero(mask)
    assert res["row"].tolist() == allowed[np.argsort(-(emb[allowed] @ q), kind="stable")][:3].tolist()


def test_filter_first_with_empty_mask_returns_nothing():
    res = DenseIndex(unit_rows(10)).filter_first(unit_rows(1)[0], np.zeros(10, dtype=bool), k=5)
    assert res.empty


def test_post_filter_doubles_k_until_enough_survive():
    emb = unit_rows(400)
    q = emb[0]
    order = np.argsort(-(emb @ q), kind="stable")
    mask = np.zeros(400, dtype=bool)
    mask[order[[150, 160, 170]]] = True  # allowed rows sit at dense ranks 151-171
    kept, k_used = DenseIndex(emb).post_filter(q, mask, n=3, k0=100, k_max=1000)
    assert k_used == 200 and kept["row"].tolist() == order[[150, 160, 170]].tolist()
    assert kept["rank"].tolist() == [1, 2, 3]


def test_post_filter_returns_short_list_at_k_max():
    emb = unit_rows(400)
    q = emb[0]
    order = np.argsort(-(emb @ q), kind="stable")
    mask = np.zeros(400, dtype=bool)
    mask[order[350]] = True  # only one allowed row, outside k_max
    kept, k_used = DenseIndex(emb).post_filter(q, mask, n=5, k0=100, k_max=300)
    assert k_used == 300 and kept.empty
    # Filter-first still finds it: the gap S5 is meant to close.
    assert DenseIndex(emb).filter_first(q, mask, k=5)["row"].tolist() == [order[350]]


# ---- retrieve() entry point (S3/S4/S5) -------------------------------------------------


def small_catalog(n: int = 40) -> pd.DataFrame:
    price = np.where(np.arange(n) % 4 == 0, np.nan, 10.0 + np.arange(n))
    return pd.DataFrame({"title": [f"item {i}" for i in range(n)], "features": [""] * n,
                         "category": ["A" if i < 20 else "B" for i in range(n)],
                         "brand": ["X"] * n, "price": price})


def test_retrieve_modes_respect_or_ignore_filters():
    cat, emb = small_catalog(), unit_rows(40)
    index, qv = DenseIndex(emb), emb[25]
    pq = ParsedQuery(semantic_query="x", category="A", max_price=25)
    allowed = set(np.flatnonzero((cat.category == "A") & (cat.price <= 25)))
    s5 = retrieve(pq, "filter_first", 200, catalog=cat, index=index, query_vec=qv)
    assert set(s5.candidates.row) == allowed and s5.n_satisfying == len(allowed)
    s4 = retrieve(pq, "post_filter", 200, catalog=cat, index=index, query_vec=qv, k0=10, k_max=40)
    assert set(s4.candidates.row) <= allowed and s4.k_used in (10, 20, 40)
    s3 = retrieve(pq, "dense", 5, catalog=cat, index=index, query_vec=qv)
    assert s3.candidates.row.tolist()[0] == 25  # ignores filters: row 25 is category B


def test_retrieve_reports_price_exclusions_and_unpriced_fallback():
    cat, emb = small_catalog(), unit_rows(40)
    pq = ParsedQuery(semantic_query="x", category="A", max_price=12)  # only rows 1, 2 qualify
    res = retrieve(pq, "filter_first", 200, catalog=cat, index=DenseIndex(emb), query_vec=emb[0], n=5)
    assert sorted(res.candidates.row) == [1, 2]
    assert res.price_excluded == 5                       # rows 0, 4, 8, 12, 16 have no price
    assert sorted(res.unpriced.row) == [0, 4, 8, 12, 16]
    assert not set(res.unpriced.row) & set(res.candidates.row)
    roomy = retrieve(pq.model_copy(update={"max_price": 100}), "filter_first", 200, catalog=cat,
                     index=DenseIndex(emb), query_vec=emb[0], n=5)
    assert roomy.unpriced.empty                          # enough priced results -> no fallback list


def test_post_filter_widens_k_until_n_survive_for_top10_eval():
    emb = unit_rows(1000, seed=7)
    q = emb[0]
    order = np.argsort(-(emb @ q), kind="stable")
    mask = np.zeros(1000, dtype=bool)
    mask[order[::30]] = True  # one allowed row every 30 dense ranks -> ~3 per 100
    kept, k_used = DenseIndex(emb).post_filter(q, mask, n=10, k0=100, k_max=1000)
    assert len(kept) >= 10 and k_used == 400  # 100 -> 200 -> 400 needed for 10 survivors
    cat = pd.DataFrame({"title": ["t"] * 1000, "features": [""] * 1000, "category": ["A"] * 1000,
                        "brand": ["X"] * 1000, "price": np.where(mask, 10.0, 99.0)})
    res = retrieve(ParsedQuery(semantic_query="x", max_price=10), "post_filter", 200, catalog=cat,
                   index=DenseIndex(emb), query_vec=q, n=10)
    assert len(res.candidates) >= 10 and res.k_used == 400


def test_retrieve_rejects_unknown_mode():
    with pytest.raises(ValueError):
        retrieve(ParsedQuery(semantic_query="x"), "magic", 5, catalog=small_catalog(),
                 index=DenseIndex(unit_rows(40)), query_vec=unit_rows(1)[0])


# ---- Phase 13.5: hybrid keyword + semantic ----------------------------------------------------------------

def test_hybrid_filter_first_blends_dense_and_keyword_scores_inside_the_mask():
    from src.retrieve import DenseIndex, hybrid_scores

    e = np.array([[1, 0], [0.8, 0.6], [0, 1], [0.9, 0.1]], dtype=np.float32)
    index = DenseIndex(e)
    q = np.array([1, 0], dtype=np.float32)
    lexical = np.array([0.0, 5.0, 9.0, 1.0])                 # keyword match prefers row 2 (masked out below)
    mask = np.array([True, True, False, True])
    dense = index.filter_first(q, mask, 3)
    assert dense["row"].tolist() == [0, 3, 1]
    assert index.hybrid_filter_first(q, lexical, mask, 3, alpha=0.0)["row"].tolist() == [0, 3, 1]
    assert index.hybrid_filter_first(q, lexical, mask, 3, alpha=1.0)["row"].tolist() == [1, 3, 0]
    mixed = index.hybrid_filter_first(q, lexical, mask, 3, alpha=0.5)
    assert 2 not in mixed["row"].tolist()                      # hard filters still win
    s = hybrid_scores(np.array([0.9, 0.5, 0.7]), np.array([0.0, 4.0, 2.0]), 0.5)
    assert s.tolist() == pytest.approx([0.5, 0.5, 0.5])



def test_hybrid_keeps_the_true_cosine_and_blends_the_unpriced_list():
    from src.retrieve import DenseIndex

    e = np.array([[1, 0], [0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
    cands = DenseIndex(e).hybrid_filter_first(np.array([1, 0], dtype=np.float32), np.array([0.0, 1.0, 5.0]),
                                              np.ones(3, dtype=bool), 3, alpha=0.5)
    assert dict(zip(cands["row"], cands["cosine"])) == pytest.approx({0: 1.0, 1: 0.8, 2: 0.6})
    catalog = pd.DataFrame({"title": ["a", "b x9", "c"], "features": ["", "", ""], "category": ["X"] * 3,
                            "brand": ["B"] * 3, "price": [np.nan, np.nan, 500.0]})
    res = retrieve(ParsedQuery(semantic_query="x9", max_price=100), "filter_first", 10, catalog=catalog,
                   index=DenseIndex(e), query_vec=np.array([1, 0], dtype=np.float32), n=5,
                   lexical_scores=np.array([0.0, 9.0, 0.0]), alpha=1.0)
    assert res.unpriced["row"].tolist()[0] == 1                          # keyword match first among unpriced
