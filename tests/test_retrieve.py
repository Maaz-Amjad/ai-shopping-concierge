"""Tests for src/retrieve.py on small synthetic data."""

import numpy as np
import pytest

from src.retrieve import BM25Retriever, DenseIndex, TfidfRetriever, tokenize, top_k

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
