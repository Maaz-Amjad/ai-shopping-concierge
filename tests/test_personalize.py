"""Tests for src/personalize.py on tiny hand-built catalogs."""

import math

import numpy as np
import pandas as pd
import pytest

from src.personalize import (Weights, build_profiles, holdout_split, item_rank, minmax, next_item_eval,
                             popularity_score, rank, rating_score, summarize_next_item, user_vector)


def toy_catalog() -> tuple[pd.DataFrame, np.ndarray]:
    catalog = pd.DataFrame({"product_id": ["A", "B", "C", "D"], "popularity": [10, 100, 10, 1]})
    e = np.array([[1, 0], [0, 1], [1, 0], [0.6, 0.8]], dtype=np.float32)
    return catalog, e


def test_minmax_and_constant_scores_give_zeros():
    assert minmax(np.array([2.0, 4.0, 3.0])).tolist() == [0.0, 1.0, 0.5]
    assert minmax(np.array([0.7, 0.7])).tolist() == [0.0, 0.0]          # spec 8.4: no division by zero
    assert len(minmax(np.array([]))) == 0


def test_popularity_score_is_log_scaled():
    assert popularity_score(np.array([0, 99]), 99).tolist() == pytest.approx([0.0, 1.0])
    assert popularity_score(np.array([9]), 99)[0] == pytest.approx(math.log(10) / math.log(100))


def test_popularity_score_is_zero_when_no_product_has_interactions():
    # 29 Oct: an imported store has no sales history (all counts 0); log1p(0) = 0 made S_pop NaN and scrambled ranks.
    assert popularity_score(np.array([0, 0]), 0).tolist() == [0.0, 0.0]


def test_cold_start_drops_personal_weight_and_renormalizes():
    w = Weights(0.7, 0.25, 0.05).cold_start()
    assert (w.sem, w.pers, w.pop) == pytest.approx((0.7 / 0.75, 0.0, 0.05 / 0.75))
    assert Weights.from_config({"ranking": {"w_sem": 0.6, "w_pers": 0.3, "w_pop": 0.1}}) == Weights(0.6, 0.3, 0.1)


def test_user_vector_mean_recency_and_empty_history():
    _, e = toy_catalog()
    v = user_vector(np.array([0, 1]), e)
    assert v == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert user_vector(np.array([], dtype=int), e) is None
    # Half-life 1 day: the item 1 day older gets weight 0.5 -> (1, 0.5) normalized.
    v = user_vector(np.array([1, 0]), e, timestamps=np.array([0, 86_400_000]), half_life_days=1)
    assert v == pytest.approx(np.array([1, 0.5]) / math.sqrt(1.25), abs=1e-6)


def test_build_profiles_skips_unknown_products():
    catalog, e = toy_catalog()
    inter = pd.DataFrame({"user_id": ["u1", "u1", "u2"], "product_id": ["A", "ZZZ", "ZZZ"], "timestamp": [1, 2, 3]})
    profiles = build_profiles(inter, catalog, e)
    assert list(profiles) == ["u1"] and profiles["u1"] == pytest.approx([1.0, 0.0])


def test_rank_hand_computed_components_and_ties():
    catalog, e = toy_catalog()
    cands = pd.DataFrame({"row": [0, 1, 3], "score": [0.9, 0.5, 0.7]})
    u = np.array([0, 1], dtype=np.float32)                               # likes B-style items
    out = rank(cands, u, Weights(0.5, 0.5, 0.0), catalog=catalog, embeddings=e).set_index("product_id")
    # s_sem: A 1, B 0, D 0.5; personal cosine A 0, B 1, D 0.8 -> s_pers A 0, B 1, D 0.8
    assert out.loc["D", "final"] == pytest.approx(0.65)
    assert out.loc["A", "final"] == out.loc["B", "final"] == pytest.approx(0.5)
    assert out["rank"].to_dict() == {"D": 1, "B": 2, "A": 3}             # tie: B (pop 100) beats A (pop 10)


def test_rank_cold_start_equals_semantic_order_when_no_popularity_weight():
    catalog, e = toy_catalog()
    cands = pd.DataFrame({"row": [0, 1, 3], "score": [0.9, 0.5, 0.7]})
    out = rank(cands, None, Weights(0.7, 0.25, 0.0), catalog=catalog, embeddings=e)
    assert out["product_id"].tolist() == ["A", "D", "B"] and (out["s_pers"] == 0).all()
    assert out["final"].tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_holdout_takes_last_interaction_per_user():
    inter = pd.DataFrame({"user_id": ["u", "u", "u", "v", "v"], "product_id": ["A", "B", "C", "A", "D"],
                          "timestamp": [3, 1, 2, 5, 5]})
    hist, held = holdout_split(inter)
    assert held.set_index("user_id")["product_id"].to_dict() == {"u": "A", "v": "D"}   # v tie: product_id
    assert len(hist) == 3


def test_item_rank_uses_popularity_then_id_tie_break():
    scores = np.array([0.5, 0.5, 0.9, 0.5])
    pop = np.array([10, 10, 1, 20])
    order = np.array([0, 1, 2, 3])
    cands = np.array([0, 1, 2, 3])
    assert item_rank(scores, 0, pop, order, cands) == 3      # behind 2 (score) and 3 (popularity)
    assert item_rank(scores, 1, pop, order, cands) == 4      # behind 0 on product_id
    assert item_rank(scores, 1, pop, order, np.array([1, 2])) == 2


def test_next_item_eval_profile_finds_the_similar_item():
    catalog, e = toy_catalog()
    # u saw A (direction [1,0]); held-out item C is identical in direction -> profile rank 1.
    inter = pd.DataFrame({"user_id": ["u", "u"], "product_id": ["A", "C"], "timestamp": [1, 2]})
    per_user = next_item_eval(inter, catalog, e, Weights(0.7, 0.25, 0.05), k=2)
    r = per_user.set_index("method")["rank"].to_dict()
    assert r["profile"] == 1 and r["popularity"] == 2         # unseen: B (pop 100), C (10), D (1)
    assert per_user.set_index("method").loc["profile", "ndcg@2"] == 1.0
    s = summarize_next_item(per_user, k=2).set_index("method")
    assert s.loc["popularity", "hit@2"] == 1.0 and s.loc["popularity", "ndcg@2"] == pytest.approx(1 / math.log2(3))


def test_pick_users_is_seeded_and_respects_minimum():
    from src.personalize import pick_users

    inter = pd.DataFrame({"user_id": ["a"] * 10 + ["b"] * 12 + ["c"] * 3 + ["d"] * 10,
                          "product_id": [str(i) for i in range(35)], "timestamp": range(35)})
    users = pick_users(inter, n=2, min_interactions=10, seed=42)
    assert len(users) == 2 and set(users) <= {"a", "b", "d"} and users == pick_users(inter, 2, 10, 42)


def test_next_item_popularity_counts_only_history_not_held_out_items():
    # Verifier 28 Oct: catalog popularity includes the held-out interactions (leak).
    catalog, e = toy_catalog()                                          # catalog popularity: B = 100 (leaky)
    inter = pd.DataFrame({"user_id": ["u", "u", "x", "x"], "product_id": ["A", "B", "C", "D"],
                          "timestamp": [1, 2, 1, 2]})
    per_user = next_item_eval(inter, catalog, e, Weights(0.7, 0.25, 0.05), k=10)
    r = per_user[per_user.user_id == "u"].set_index("method")["rank"]
    # History counts: A 1, C 1, B 0, D 0. u saw A; unseen B, C, D -> C first, then B before D on product_id.
    assert r["popularity"] == 2


# ---- Phase 13.1: ratings as a fourth ranking term -------------------------------------------------------

def test_rating_score_is_a_bayesian_average_and_missing_is_neutral():
    out = rating_score(np.array([5.0, 4.5, np.nan]), np.array([1, 1000, 0]), mean=4.0, prior=20)
    assert out.tolist() == pytest.approx([(80 + 5) / 21, (80 + 4500) / 1020, 4.0])   # 1 rating barely moves it


def test_weights_with_rating_from_config_and_cold_start():
    w = Weights.from_config({"ranking": {"w_sem": 0.6, "w_pers": 0.25, "w_pop": 0.05, "w_rat": 0.1}})
    assert w == Weights(0.6, 0.25, 0.05, 0.1)
    assert Weights.from_config({"ranking": {"w_sem": 0.7, "w_pers": 0.25, "w_pop": 0.05}}).rat == 0.0
    cold = w.cold_start()
    assert cold.pers == 0 and cold.sem + cold.pop + cold.rat == pytest.approx(1.0)
    assert cold.rat == pytest.approx(0.1 / 0.75)


def test_rank_rating_term_hand_computed_and_zero_weight_changes_nothing():
    catalog, e = toy_catalog()
    catalog = catalog.assign(avg_rating=[5.0, 3.0, 4.0, np.nan], rating_count=[100, 100, 1, 0])
    cands = pd.DataFrame({"row": [0, 1, 3], "score": [0.5, 0.9, 0.7]})       # s_sem A 0, B 1, D 0.5
    # catalog mean 4.0; Bayesian A 4.83, B 3.17, D 4.0 (no ratings) -> s_rat A 1, B 0, D 0.5
    out = rank(cands, None, Weights(0.4, 0.0, 0.0, 0.6), catalog=catalog, embeddings=e).set_index("product_id")
    assert out["s_rat"].to_dict() == pytest.approx({"A": 1.0, "B": 0.0, "D": 0.5})
    assert out["rank"].to_dict() == {"A": 1, "D": 2, "B": 3}
    plain = rank(cands, None, Weights(0.7, 0.25, 0.05), catalog=catalog, embeddings=e)
    no_ratings = rank(cands, None, Weights(0.7, 0.25, 0.05), catalog=catalog.drop(columns=["avg_rating",
                                                                                         "rating_count"]),
                      embeddings=e)
    assert plain["final"].tolist() == no_ratings["final"].tolist() and (no_ratings["s_rat"] == 0).all()


# ---- verifier findings, Phase 13.1 ----------------------------------------------------------------------

def test_production_weights_on_a_catalog_without_ratings_rank_exactly_like_the_spec():
    catalog, e = toy_catalog()                                   # no rating columns (older store imports)
    cands = pd.DataFrame({"row": [0, 1, 3], "score": [0.9, 0.85, 0.7]})
    u = np.array([0.6, 0.8], dtype=np.float32)
    for user in (None, u):
        spec = rank(cands, user, Weights(0.7, 0.25, 0.05), catalog=catalog, embeddings=e)
        prod = rank(cands, user, Weights(0.6, 0.25, 0.05, 0.1), catalog=catalog, embeddings=e)
        assert prod["product_id"].tolist() == spec["product_id"].tolist()
        assert prod["final"].tolist() == pytest.approx(spec["final"].tolist())


def test_rank_tolerates_messy_rating_columns():
    catalog, e = toy_catalog()
    cands = pd.DataFrame({"row": [0, 1, 3], "score": [0.9, 0.5, 0.7]})
    w = Weights(0.6, 0.25, 0.05, 0.1)
    no_count = catalog.assign(avg_rating=[4.5, 3.0, 4.0, np.nan])                      # no rating_count column
    assert len(rank(cands, None, w, catalog=no_count, embeddings=e)) == 3
    messy = catalog.assign(avg_rating=["4.5", "x", None, 4.0], rating_count=["10", "x", None, np.inf])
    assert len(rank(cands, None, w, catalog=messy, embeddings=e)) == 3
