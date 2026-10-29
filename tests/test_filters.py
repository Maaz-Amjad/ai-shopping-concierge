"""Tests for src/filters.py on a small hand-built catalog."""

import numpy as np
import pandas as pd

from src.filters import build_mask, exclusion_mask, relaxation_suggestions, unpriced_mask
from src.parse import ParsedQuery

DF = pd.DataFrame({
    "title": ["Ice maker steel", "Ice maker white", "Range hood", "Ice maker black", "Ice maker blue"],
    "features": ["stainless steel body", "", "", "", ""],
    "category": ["Ice Makers", "Ice Makers", "Range Hoods", "Ice Makers", "Ice Makers"],
    "brand": ["FRIGIDAIRE", "Frigidaire", "GE", None, "GE"],
    "price": [100.0, 150.0, 80.0, np.nan, 150.01],
})


def q(**kw) -> ParsedQuery:
    return ParsedQuery(semantic_query="x", **kw)


def rows(mask) -> list[int]:
    return np.flatnonzero(mask).tolist()


def test_no_constraints_keeps_everything_including_null_price():
    assert rows(build_mask(DF, q())) == [0, 1, 2, 3, 4]


def test_price_bounds_are_inclusive_and_null_price_excluded():
    assert rows(build_mask(DF, q(max_price=150))) == [0, 1, 2]       # 150.01 and NaN excluded
    assert rows(build_mask(DF, q(min_price=150))) == [1, 4]
    assert rows(build_mask(DF, q(min_price=90, max_price=150))) == [0, 1]


def test_category_and_case_insensitive_brand():
    assert rows(build_mask(DF, q(category="Ice Makers", brand="FRIGIDAIRE"))) == [0, 1]
    assert rows(build_mask(DF, q(brand="ge"))) == [2, 4]


def test_exclusions_check_title_and_features():
    assert rows(exclusion_mask(DF, ["stainless"])) == [1, 2, 3, 4]
    assert rows(build_mask(DF, q(exclude_terms=["white", "steel"]))) == [2, 3, 4]
    assert rows(build_mask(DF, q(exclude_terms=["white"]), use_exclusions=False)) == [0, 1, 2, 3, 4]


def test_unpriced_mask_only_under_a_price_filter():
    assert rows(unpriced_mask(DF, q(category="Ice Makers"))) == []
    assert rows(unpriced_mask(DF, q(category="Ice Makers", max_price=50))) == [3]
    assert rows(unpriced_mask(DF, q(category="Range Hoods", max_price=50))) == []


def test_relaxation_suggestions_rank_by_products_restored():
    # GE ice maker under $120: nothing matches.
    pq = q(category="Ice Makers", brand="GE", max_price=120)
    assert build_mask(DF, pq).sum() == 0
    # drop price -> row 4; drop brand -> rows 0; drop category -> row 2 (GE hood at $80)
    assert relaxation_suggestions(DF, pq) == [("brand", 1), ("category", 1), ("price", 1)]
    pq2 = q(category="Ice Makers", max_price=90)
    assert relaxation_suggestions(DF, pq2) == [("price", 4), ("category", 1)]



def test_out_of_stock_products_are_never_candidates_unknown_stock_is_kept():
    df = pd.DataFrame({"title": ["a", "b", "c"], "features": ["", "", ""], "price": [1.0, 2.0, np.nan],
                       "category": ["X", "X", "X"], "brand": ["B", "B", "B"], "in_stock": [True, False, None]})
    assert rows(build_mask(df, q())) == [0, 2]
    assert rows(unpriced_mask(df, q(max_price=5))) == [2]


def test_has_hard_limits():
    from src.filters import has_hard_limits

    assert not has_hard_limits(q(exclude_terms=["steel"]))               # exclusions are not a stated limit here
    assert all(has_hard_limits(q(**kw)) for kw in ({"max_price": 5}, {"min_price": 1}, {"category": "X"},
                                                    {"brand": "GE"}))
