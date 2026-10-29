"""Load, clean and persist the product catalog and interactions (tech spec section 5).

Run `python -m src.data` to rebuild data/processed/ from data/raw/ using config.yaml.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import ROOT, load_config

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEANING_LOG_PATH = ROOT / "eval" / "results" / "cleaning_log.json"

MIN_TEXT_CHARS = 20          # spec 5.2: description + features combined
MIN_CATEGORY_SIZE = 30       # spec 5.1: smaller categories map to Other
PRICE_OUTLIER_QUANTILE = 0.999
MAX_TEXT_CHARS = 2000        # bge-small truncates at 512 tokens anyway; keep text bounded
OTHER = "Other"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_PRICE_RE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)")


def clean_text(text: str | None) -> str:
    """Strip HTML tags and entities and normalize whitespace."""
    if not text:
        return ""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", str(text)))).strip()


def parse_price(raw: object) -> float | None:
    """Parse '$12.99' or '12.99' to a float; anything else (None, ranges, 0) -> None."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    match = _PRICE_RE.fullmatch(str(raw).replace(",", "").strip())
    if not match:
        return None
    value = float(match.group(1))
    return value if value > 0 else None


def extract_category(categories: list[str] | None) -> str | None:
    """Deepest category level up to level 3 (index 2); None when the list is empty."""
    if not categories:
        return None
    return clean_text(categories[min(len(categories), 3) - 1]) or None


def load_raw_meta(path: Path) -> pd.DataFrame:
    """Read the Amazon Reviews 2023 metadata JSONL into the fields we need."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "product_id": d.get("parent_asin"),
                "title": clean_text(d.get("title")),
                "brand": clean_text(d.get("store")) or None,
                "category": extract_category(d.get("categories")),
                "price": parse_price(d.get("price")),
                "price_raw_present": d.get("price") not in (None, "", "None"),
                "description": clean_text(" ".join(d.get("description") or [])),
                "features": "; ".join(t for t in map(clean_text, d.get("features") or []) if t),
                "avg_rating": d.get("average_rating"),          # Phase 13.1: rating signal (not embedded)
                "rating_count": d.get("rating_number") or 0,
            })
    return pd.DataFrame(rows)


def load_raw_interactions(path: Path) -> pd.DataFrame:
    """Read the rating-only CSV into user_id, product_id, rating, timestamp (ms)."""
    df = pd.read_csv(path)
    return df.rename(columns={"parent_asin": "product_id"})[["user_id", "product_id", "rating", "timestamp"]]


def clean_catalog(meta: pd.DataFrame, interactions: pd.DataFrame, max_items: int) -> tuple[pd.DataFrame, dict]:
    """Apply the spec 5.2 cleaning rules, keep the `max_items` most-interacted products.

    Returns the catalog and a log of how many rows each rule removed or changed.
    """
    log: dict[str, int] = {"raw_items": len(meta)}
    df = meta.copy()

    before = len(df)
    df = df[df["product_id"].notna()].drop_duplicates("product_id")
    log["dropped_duplicate_or_missing_id"] = before - len(df)

    before = len(df)
    df = df[df["title"].str.len() > 0]
    log["dropped_empty_title"] = before - len(df)

    before = len(df)
    text_len = df["description"].str.len() + df["features"].str.len()
    df = df[text_len >= MIN_TEXT_CHARS]
    log["dropped_short_text"] = before - len(df)

    popularity = interactions["product_id"].value_counts()
    df = df.assign(popularity=df["product_id"].map(popularity).fillna(0).astype(int))
    # Most interacted first; product_id breaks ties so the order is deterministic.
    df = df.sort_values(["popularity", "product_id"], ascending=[False, True])

    before = len(df)
    df = df.drop_duplicates("title", keep="first")  # keeps the most-interacted copy
    log["dropped_duplicate_title"] = before - len(df)

    log["eligible_items"] = len(df)
    df = df.head(max_items).reset_index(drop=True)
    log["kept_top_n"] = len(df)

    log["price_invalid_set_null"] = int((df["price_raw_present"] & df["price"].isna()).sum())
    df = df.drop(columns="price_raw_present")
    cap = df["price"].quantile(PRICE_OUTLIER_QUANTILE)
    outliers = df["price"] > cap
    df.loc[outliers, "price"] = np.nan
    log["price_outliers_set_null"] = int(outliers.sum())
    log["price_outlier_cap"] = round(float(cap), 2)
    log["price_missing"] = int(df["price"].isna().sum())

    counts = df["category"].value_counts()
    small = df["category"].isna() | df["category"].map(counts).lt(MIN_CATEGORY_SIZE)
    df.loc[small, "category"] = OTHER
    log["items_mapped_to_other_category"] = int(small.sum())
    log["n_categories"] = int(df["category"].nunique())
    log["brand_missing"] = int(df["brand"].isna().sum())

    df["product_text"] = df.apply(build_product_text, axis=1)
    return df, log


def build_product_text(row: pd.Series) -> str:
    """Spec 5.3: '{title}. Brand: {brand}. Category: {category}. Features: {features}. {description}'.

    Missing parts are left out. Price is not embedded (filters handle it).
    """
    parts = [row["title"]]
    if isinstance(row["brand"], str) and row["brand"]:
        parts.append(f"Brand: {row['brand']}")
    if isinstance(row["category"], str) and row["category"] != OTHER:
        parts.append(f"Category: {row['category']}")
    if row["features"]:
        parts.append(f"Features: {row['features']}")
    if row["description"]:
        parts.append(row["description"])
    return ". ".join(p.rstrip(". ") for p in parts)[:MAX_TEXT_CHARS]


def filter_interactions(interactions: pd.DataFrame, product_ids: pd.Series, min_user_interactions: int) -> tuple[pd.DataFrame, dict]:
    """Keep interactions on catalog products by users with >= `min_user_interactions` of them."""
    log = {"raw_interactions": len(interactions)}
    df = interactions[interactions["product_id"].isin(set(product_ids))]
    df = df.drop_duplicates(["user_id", "product_id", "timestamp"])
    log["catalog_interactions"] = len(df)
    counts = df["user_id"].value_counts()
    df = df[df["user_id"].map(counts) >= min_user_interactions]
    df = df.sort_values(["user_id", "timestamp", "product_id"]).reset_index(drop=True)
    log["kept_interactions"] = len(df)
    log["kept_users"] = int(df["user_id"].nunique())
    log["users_with_10_plus"] = int((df["user_id"].value_counts() >= 10).sum())
    return df, log


def build(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rebuild catalog and interactions from raw files and persist them with the cleaning log."""
    cfg = cfg or load_config()
    category = cfg["dataset"]["category"]
    meta = load_raw_meta(RAW_DIR / f"meta_{category}.jsonl")
    raw_inter = load_raw_interactions(RAW_DIR / f"{category}_rating_only.csv")
    catalog, log = clean_catalog(meta, raw_inter, cfg["dataset"]["max_items"])
    inter, inter_log = filter_interactions(raw_inter, catalog["product_id"], cfg["dataset"]["min_user_interactions"])
    log = {"dataset": category, **log, **inter_log}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    catalog.to_pickle(PROCESSED_DIR / "catalog.pkl")
    inter.to_pickle(PROCESSED_DIR / "interactions.pkl")
    CLEANING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return catalog, inter, log


def load_catalog() -> pd.DataFrame:
    return pd.read_pickle(PROCESSED_DIR / "catalog.pkl")


def load_interactions() -> pd.DataFrame:
    return pd.read_pickle(PROCESSED_DIR / "interactions.pkl")


if __name__ == "__main__":
    _, _, cleaning_log = build()
    print(json.dumps(cleaning_log, indent=2))
