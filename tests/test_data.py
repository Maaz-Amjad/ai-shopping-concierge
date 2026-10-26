"""Tests for src/data.py on small hand-built data (no raw files needed)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import data
from src.data import (OTHER, build_product_text, clean_catalog, clean_text, extract_category,
                      filter_interactions, load_raw_meta, parse_price)

LONG = "a description that is long enough"


def item(pid, title=None, brand="Acme", category="Ice Makers", price=10.0, description=LONG,
         features="", price_raw_present=None):
    return {"product_id": pid, "title": title or f"Product {pid}", "brand": brand, "category": category,
            "price": price, "description": description, "features": features,
            "price_raw_present": price is not None if price_raw_present is None else price_raw_present}


def inter(pid_counts: dict[str, int]) -> pd.DataFrame:
    rows = [{"user_id": f"u{i}", "product_id": pid, "rating": 5.0, "timestamp": i}
            for pid, n in pid_counts.items() for i in range(n)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("raw, expected", [
    ("$12.99", 12.99), ("12.99", 12.99), ("$1,299.00", 1299.0), (None, None), ("None", None),
    ("$10 - $20", None), ("0", None), ("-5", None), (float("nan"), None),
])
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


def test_clean_text_strips_html_and_whitespace():
    # Tags become spaces so adjacent words never fuse ("Quiet</b>fast" -> "Quiet fast").
    assert clean_text("<b>Quiet</b>&amp;\n  fast <br/>") == "Quiet & fast"
    assert clean_text("Quiet</b>fast") == "Quiet fast"
    assert clean_text(None) == ""


def test_extract_category_uses_deepest_level_up_to_three():
    assert extract_category(["Appliances", "Parts & Accessories", "Dryer Parts", "Belts"]) == "Dryer Parts"
    assert extract_category(["Appliances", "Dishwashers"]) == "Dishwashers"
    assert extract_category([]) is None


def test_load_raw_meta_maps_fields(tmp_path):
    rec = {"parent_asin": "P1", "title": " Ice <i>Maker</i> ", "store": "ROVSUN", "price": "$99.50",
           "categories": ["Appliances", "Refrigerators", "Ice Makers"],
           "description": ["Makes ice."], "features": ["Quiet", " Portable "]}
    path = tmp_path / "meta.jsonl"
    path.write_text(json.dumps(rec) + "\n" + json.dumps({"parent_asin": "P2", "store": "", "price": None}) + "\n")
    df = load_raw_meta(path)
    assert df.loc[0, ["product_id", "title", "brand", "category", "price", "features"]].tolist() == \
        ["P1", "Ice Maker", "ROVSUN", "Ice Makers", 99.5, "Quiet; Portable"]
    assert pd.isna(df.loc[1, "brand"]) and pd.isna(df.loc[1, "price"]) and not df.loc[1, "price_raw_present"]


def make_catalog_inputs():
    items = [item(f"P{i:02d}") for i in range(40)]                    # 40 Ice Makers, price 10
    items += [item("DUP")] + [item("DUP", title="duplicate id")]       # duplicate product_id
    items += [item("NOTITLE", title=" ")]                               # empty title -> dropped
    items[-1]["title"] = ""
    items += [item("SHORT", description="too short", features="")]   # < 20 chars -> dropped
    items += [item("SAMETITLE_POP", title="Same title"), item("SAMETITLE_RARE", title="Same title")]
    items += [item("RARECAT", category="Kegerators"), item("NOCAT", category=None)]
    items += [item("NOPRICE", price=None), item("BADPRICE", price=None, price_raw_present=True)]
    items += [item("EXPENSIVE", price=1_000_000.0)]
    meta = pd.DataFrame(items)
    counts = {"SAMETITLE_POP": 5, "SAMETITLE_RARE": 1, "EXPENSIVE": 3}
    return meta, inter(counts)


def test_clean_catalog_rules_and_log():
    meta, interactions = make_catalog_inputs()
    cat, log = clean_catalog(meta, interactions, max_items=1000)
    ids = set(cat["product_id"])
    assert log["dropped_duplicate_or_missing_id"] == 1
    dup_rows = cat[cat["product_id"] == "DUP"]
    assert len(dup_rows) == 1 and dup_rows["title"].item() == "Product DUP"  # first occurrence kept
    assert log["dropped_empty_title"] == 1 and "NOTITLE" not in ids
    assert log["dropped_short_text"] == 1 and "SHORT" not in ids
    assert log["dropped_duplicate_title"] == 1
    assert "SAMETITLE_POP" in ids and "SAMETITLE_RARE" not in ids  # keeps the most-interacted copy
    # Small or missing categories map to Other; Ice Makers (>=30 items) is kept.
    assert cat.set_index("product_id").loc[["RARECAT", "NOCAT"], "category"].tolist() == [OTHER, OTHER]
    assert set(cat["category"]) == {"Ice Makers", OTHER}
    assert log["items_mapped_to_other_category"] == 2
    # Prices: missing stays null; invalid raw counted; extreme outlier set to null.
    assert log["price_invalid_set_null"] == 1
    assert pd.isna(cat.set_index("product_id").loc["EXPENSIVE", "price"])
    assert log["price_outliers_set_null"] == 1
    assert log["price_missing"] == 3  # NOPRICE, BADPRICE, EXPENSIVE
    assert cat.set_index("product_id").loc["SAMETITLE_POP", "popularity"] == 5


def test_clean_catalog_keeps_top_n_by_popularity_deterministically():
    meta = pd.DataFrame([item(p) for p in ["A", "B", "C", "D"]])
    cat, log = clean_catalog(meta, inter({"C": 3, "A": 1}), max_items=3)
    assert cat["product_id"].tolist() == ["C", "A", "B"]  # ties (B, D at 0) broken by product_id
    assert log["kept_top_n"] == 3 and log["eligible_items"] == 4


def test_build_product_text_template_and_missing_parts():
    row = pd.Series(item("P", title="Ice Maker", brand="ROVSUN", category="Ice Makers",
                         features="Quiet; Portable", description="Makes ice fast."))
    assert build_product_text(row) == \
        "Ice Maker. Brand: ROVSUN. Category: Ice Makers. Features: Quiet; Portable. Makes ice fast"
    bare = pd.Series(item("P", title="Belt", brand=None, category=OTHER, description="Fits dryers"))
    assert build_product_text(bare) == "Belt. Fits dryers"
    assert "price" not in build_product_text(row).lower()


def test_filter_interactions_restricts_to_catalog_and_active_users():
    rows = [("u1", f"P{i}", i) for i in range(5)] + [("u2", "P0", 1), ("u2", "P1", 2)] \
        + [("u3", "X", 1)] * 6 + [("u1", "P0", 0)]  # last row duplicates u1's first interaction
    df = pd.DataFrame(rows, columns=["user_id", "product_id", "timestamp"]).assign(rating=5.0)
    kept, log = filter_interactions(df, pd.Series([f"P{i}" for i in range(5)]), min_user_interactions=5)
    assert set(kept["user_id"]) == {"u1"} and len(kept) == 5
    assert kept["timestamp"].is_monotonic_increasing
    assert log["kept_users"] == 1 and log["catalog_interactions"] == 7


def test_build_writes_outputs(tmp_path, monkeypatch):
    meta, interactions = make_catalog_inputs()
    monkeypatch.setattr(data, "load_raw_meta", lambda p: meta)
    monkeypatch.setattr(data, "load_raw_interactions", lambda p: interactions)
    monkeypatch.setattr(data, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(data, "CLEANING_LOG_PATH", tmp_path / "log.json")
    cfg = {"dataset": {"category": "X", "max_items": 10, "min_user_interactions": 1}}
    catalog, _, log = data.build(cfg)
    assert len(catalog) == 10 and (tmp_path / "catalog.pkl").exists()
    assert json.loads((tmp_path / "log.json").read_text())["kept_top_n"] == 10
    assert np.isnan(catalog["price"]).sum() == log["price_missing"]
