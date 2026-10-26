"""Tests for src/evaluate.py against small hand-computed examples."""

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (bootstrap_ci, build_pool, constraint_adherence, dcg, has_hard_gold, latency_summary,
                          micro_f1, ndcg_at_k, paired_bootstrap, parse_counts, precision_at_k, ranking_metrics,
                          recall_at_k, satisfies_gold, to_parsed_query, violates_exclusion)


def gold_row(**kw) -> pd.Series:
    base = {"gold_max_price": np.nan, "gold_min_price": np.nan, "gold_category": np.nan,
            "gold_brand": np.nan, "gold_exclude": np.nan}
    return pd.Series({**base, **kw})


# ---- ranking metrics ---------------------------------------------------------------------------


def test_precision_and_recall_hand_computed():
    grades = [2, 0, 1]
    assert precision_at_k(grades, 3) == pytest.approx(2 / 3)
    assert precision_at_k(grades, 5) == pytest.approx(2 / 5)      # short list still divides by K
    assert precision_at_k([], 5) == 0.0
    assert recall_at_k(grades, n_relevant_pool=3, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(grades, n_relevant_pool=4, k=1) == pytest.approx(1 / 4)
    assert math.isnan(recall_at_k(grades, n_relevant_pool=0, k=3))


def test_ndcg_hand_computed():
    # DCG = (2^2-1)/log2(2) + 0 + (2^1-1)/log2(4) = 3 + 0.5 = 3.5
    assert dcg([2, 0, 1], 3) == pytest.approx(3.5)
    # Ideal from pool [2,1,1,0]: 3 + 1/log2(3) + 1/log2(4) = 4.130930
    assert ndcg_at_k([2, 0, 1], [2, 1, 1, 0], 3) == pytest.approx(3.5 / 4.130930, abs=1e-6)
    assert ndcg_at_k([2, 1, 1], [2, 1, 1, 0], 3) == pytest.approx(1.0)
    assert ndcg_at_k([], [2, 1], 5) == 0.0
    assert math.isnan(ndcg_at_k([0, 0], [0, 0, 0], 5))           # nothing relevant in the pool


def test_ranking_metrics_treats_unjudged_as_zero_and_handles_empty_runs():
    qrels = pd.DataFrame({"query_id": ["q1", "q1", "q1", "q2"], "product_id": ["a", "b", "c", "x"],
                          "grade": [2, 1, 0, 0]})
    runs = pd.DataFrame({"system": ["S1"] * 3 + ["S5"],
                         "query_id": ["q1", "q1", "q2", "q1"],
                         "rank": [1, 2, 1, 1],
                         "product_id": ["zzz", "a", "x", "b"]})       # zzz is unjudged
    m = ranking_metrics(runs, qrels, ks=(5,)).set_index(["system", "query_id"])
    assert m.loc[("S1", "q1"), "P@5"] == pytest.approx(1 / 5)
    assert m.loc[("S1", "q1"), "R@5"] == pytest.approx(1 / 2)
    # S1 q1 grades [0, 2]: DCG = 3/log2(3); ideal [2, 1] = 3 + 1/log2(3)
    assert m.loc[("S1", "q1"), "NDCG@5"] == pytest.approx((3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    assert math.isnan(m.loc[("S1", "q2"), "R@5"]) and math.isnan(m.loc[("S1", "q2"), "NDCG@5"])
    assert m.loc[("S5", "q2"), "P@5"] == 0.0                        # S5 returned nothing for q2
    assert m.loc[("S5", "q1"), "R@5"] == pytest.approx(1 / 2)


def test_build_pool_is_union_of_top_depth():
    runs = pd.DataFrame({"query_id": ["q1"] * 5, "system": ["S1", "S1", "S1", "S3", "S3"],
                         "rank": [1, 2, 3, 1, 2], "product_id": ["a", "b", "c", "b", "d"]})
    pool = build_pool(runs, depth=2)
    assert pool["product_id"].tolist() == ["a", "b", "d"]


# ---- constraint adherence ----------------------------------------------------------------------


def items_frame() -> pd.DataFrame:
    return pd.DataFrame({"price": [10.0, np.nan, 15.0, 30.0],
                         "category": ["Ice Makers"] * 4,
                         "brand": ["GE", "GE", "ge", "GE"],
                         "title": ["clear ice", "stainless ice maker", "white ice maker", "x"],
                         "features": ["", "", "Stainless trim", None]})


def test_satisfies_gold_null_price_violates_and_brand_is_case_insensitive():
    items, gold = items_frame(), gold_row(gold_max_price=20, gold_brand="GE")
    assert [satisfies_gold(r, gold) for _, r in items.iterrows()] == [True, False, True, False]
    assert constraint_adherence(items, gold, 4) == pytest.approx(0.5)
    assert constraint_adherence(items, gold, 1) == pytest.approx(1.0)
    assert math.isnan(constraint_adherence(items.iloc[:0], gold, 5))
    assert satisfies_gold(items.iloc[3], gold_row(gold_min_price=30))       # boundary is inclusive
    assert not satisfies_gold(items.iloc[0], gold_row(gold_category="Freezers"))
    assert has_hard_gold(gold) and not has_hard_gold(gold_row(gold_exclude="stainless"))


def test_violates_exclusion_checks_title_and_features():
    items, gold = items_frame(), gold_row(gold_exclude="stainless")
    assert [violates_exclusion(r, gold) for _, r in items.iterrows()] == [False, True, True, False]
    assert not violates_exclusion(items.iloc[1], gold_row())


def test_violates_exclusion_treats_missing_features_as_empty():
    item = pd.Series({"title": "steel rack", "features": np.nan})
    assert not violates_exclusion(item, gold_row(gold_exclude="an"))   # NaN must not become the text "nan"


# ---- parse F1 ----------------------------------------------------------------------------------


def test_parse_counts_and_micro_f1_hand_computed():
    gold = pd.DataFrame({"query_id": ["q1", "q2", "q3", "q4", "q5", "q6"],
                         "gold_max_price": [25, np.nan, np.nan, np.nan, 40, np.nan],
                         "gold_min_price": [np.nan, np.nan, np.nan, np.nan, 15, np.nan],
                         "gold_category": [np.nan, "Ice Makers", np.nan, np.nan, np.nan, np.nan],
                         "gold_brand": [np.nan, np.nan, "GE", np.nan, np.nan, np.nan],
                         "gold_exclude": [np.nan] * 5 + ["samsung"]})
    pred = pd.DataFrame({"query_id": ["q1", "q2", "q3", "q4", "q5", "q6"],
                         "max_price": [25.0, None, None, None, 40.0, None],
                         "min_price": [None, None, None, None, 20.0, None],
                         "category": [None, None, None, "Ice Makers", None, None],
                         "brand": [None, None, "ge", None, None, None],
                         "exclude_terms": [[], [], [], [], [], ["Samsung"]]})
    counts = parse_counts(pred, gold).set_index("field")
    assert counts.loc["max_price"].tolist() == [2, 0, 0]
    assert counts.loc["min_price"].tolist() == [0, 1, 1]           # wrong value: one FP and one FN
    assert counts.loc["category"].tolist() == [0, 1, 1]            # q2 missed (FN), q4 invented (FP)
    assert counts.loc["brand"].tolist() == [1, 0, 0]
    assert counts.loc["exclude_terms"].tolist() == [1, 0, 0]
    hard = micro_f1(counts.loc[["max_price", "min_price", "category", "brand"]].reset_index())
    # TP 3, FP 2, FN 2 -> P = R = F1 = 0.6
    assert (hard["tp"], hard["fp"], hard["fn"]) == (3, 2, 2)
    assert hard["f1"] == pytest.approx(0.6)


def test_micro_f1_perfect_and_empty():
    assert micro_f1(pd.DataFrame({"tp": [4], "fp": [0], "fn": [0]}))["f1"] == 1.0
    assert micro_f1(pd.DataFrame({"tp": [0], "fp": [0], "fn": [0]}))["f1"] == 1.0
    assert micro_f1(pd.DataFrame({"tp": [0], "fp": [2], "fn": [1]}))["f1"] == 0.0


# ---- statistics --------------------------------------------------------------------------------


def test_bootstrap_ci_constant_values_and_determinism():
    assert bootstrap_ci([0.4] * 10) == pytest.approx((0.4, 0.4, 0.4))
    mean, lo, hi = bootstrap_ci([0, 1, 0, 1, np.nan], n_resamples=1000, seed=42)
    assert mean == 0.5 and 0 <= lo < 0.5 < hi <= 1
    assert bootstrap_ci([0, 1, 0, 1], seed=42) == bootstrap_ci([0, 1, 0, 1], seed=42)
    assert all(math.isnan(x) for x in bootstrap_ci([np.nan, np.nan]))


def test_paired_bootstrap_constant_difference_and_nan_pairs():
    a = [0.1, 0.5, 0.3, np.nan]
    b = [0.2, 0.6, 0.4, 0.9]
    assert paired_bootstrap(a, b) == pytest.approx((0.1, 0.1, 0.1))


def test_latency_summary():
    s = latency_summary(list(range(1, 101)))
    assert s["median_ms"] == 50.5 and s["p95_ms"] == pytest.approx(95.05)


# ---- harness helpers ----------------------------------------------------------------------------


def test_to_parsed_query_maps_nan_to_none():
    row = pd.Series({"semantic_query": "ice maker", "max_price": np.nan, "min_price": 10.0,
                     "category": None, "brand": np.nan, "exclude_terms": ["white"]})
    pq = to_parsed_query(row)
    assert pq.max_price is None and pq.min_price == 10.0 and pq.brand is None and pq.exclude_terms == ["white"]
