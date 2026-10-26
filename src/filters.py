"""Deterministic hard filters (spec section 8): price, category, brand, plus optional exclusions.

The LLM never filters: these masks are the only place constraints are enforced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.parse import ParsedQuery


def _brand_key(series: pd.Series) -> pd.Series:
    return series.fillna("").str.casefold()


def price_mask(df: pd.DataFrame, pq: ParsedQuery) -> np.ndarray:
    """Price bounds; a null price fails whenever a bound is active (missing-price policy)."""
    m = np.ones(len(df), dtype=bool)
    if pq.max_price is not None:
        m &= df["price"].le(pq.max_price).fillna(False).to_numpy()
    if pq.min_price is not None:
        m &= df["price"].ge(pq.min_price).fillna(False).to_numpy()
    return m


def non_price_mask(df: pd.DataFrame, pq: ParsedQuery, use_exclusions: bool = True) -> np.ndarray:
    """Category, brand (case-insensitive: catalog spellings vary) and exclusion terms."""
    m = np.ones(len(df), dtype=bool)
    if pq.category:
        m &= (df["category"] == pq.category).to_numpy()
    if pq.brand:
        m &= (_brand_key(df["brand"]) == pq.brand.casefold()).to_numpy()
    if use_exclusions and pq.exclude_terms:
        m &= exclusion_mask(df, pq.exclude_terms)
    return m


def exclusion_mask(df: pd.DataFrame, terms: list[str]) -> np.ndarray:
    """False where title or features contain any excluded term (case-insensitive substring, spec 8.3)."""
    text = (df["title"].fillna("") + " " + df["features"].fillna("")).str.lower()
    m = np.ones(len(df), dtype=bool)
    for term in terms:
        m &= ~text.str.contains(term.lower(), regex=False).to_numpy()
    return m


def build_mask(df: pd.DataFrame, pq: ParsedQuery, use_exclusions: bool = True) -> np.ndarray:
    """All hard constraints combined (spec 8.1)."""
    return price_mask(df, pq) & non_price_mask(df, pq, use_exclusions)


def has_price_filter(pq: ParsedQuery) -> bool:
    return pq.max_price is not None or pq.min_price is not None


def unpriced_mask(df: pd.DataFrame, pq: ParsedQuery, use_exclusions: bool = True) -> np.ndarray:
    """Items that pass every non-price constraint but have no price, under an active price filter.

    These feed the separate "price not listed" list (approved deviation); they are never ranked
    recommendations and are never counted in the metrics.
    """
    if not has_price_filter(pq):
        return np.zeros(len(df), dtype=bool)
    return non_price_mask(df, pq, use_exclusions) & df["price"].isna().to_numpy()


def relaxation_suggestions(df: pd.DataFrame, pq: ParsedQuery) -> list[tuple[str, int]]:
    """For each active constraint, how many products remain if only that one is removed (spec 9).

    Sorted by the number of products restored, most first. Price bounds count as one constraint.
    """
    options: dict[str, ParsedQuery] = {}
    if has_price_filter(pq):
        options["price"] = pq.model_copy(update={"max_price": None, "min_price": None})
    if pq.category:
        options["category"] = pq.model_copy(update={"category": None})
    if pq.brand:
        options["brand"] = pq.model_copy(update={"brand": None})
    if pq.exclude_terms:
        options["exclusions"] = pq.model_copy(update={"exclude_terms": []})
    counts = [(name, int(build_mask(df, relaxed).sum())) for name, relaxed in options.items()]
    return sorted(counts, key=lambda item: (-item[1], item[0]))
