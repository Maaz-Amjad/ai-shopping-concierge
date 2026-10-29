"""Store connector: presets, custom mapping, cleaning, and a build that load_resources can read back."""

import json

import numpy as np
import pandas as pd
import pytest

from src import store
from src.store import import_feed, read_feed

SHOPIFY_CSV = """Handle,Title,Body (HTML),Vendor,Product Category,Type,Tags,Variant Price
steel-kettle,Steel Kettle,<p>1.7 L <b>fast</b> boil</p>,Brewly,Home & Garden > Kitchen > Kettles,,"steel, cordless",39.00
steel-kettle,,,,,,,41.00
glass-kettle,Glass Kettle,<p>Blue LED</p>,Brewly,,Kettles,,
toaster-2,2-Slice Toaster,Wide slots,,,Toasters,,"1,299.50"
"""

WOO_CSV = """ID,SKU,Name,Description,Short description,Regular price,Sale price,Categories,Brands
11,K-1,Travel Kettle,Small and light,Folding handle,25,19.99,"Kitchen > Kettles, Gifts",Brewly
12,,Milk Frother,Handheld,,15,,Kitchen > Coffee,
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_shopify_csv_variants_html_category_and_prices(tmp_path):
    cat, log = import_feed(read_feed(write(tmp_path, "s.csv", SHOPIFY_CSV)), "shopify")
    assert cat["product_id"].tolist() == ["steel-kettle", "glass-kettle", "toaster-2"]
    assert log["dropped_no_id_or_title"] == 1 and log["products"] == 3       # variant row without a title
    first = cat.iloc[0]
    assert first["description"] == "1.7 L fast boil" and first["category"] == "Kettles" and first["price"] == 39.0
    assert cat.iloc[1]["category"] == "Kettles"                               # falls back to Type
    assert np.isnan(cat.iloc[1]["price"]) and log["price_missing"] == 1       # missing price stays null
    assert cat.iloc[2]["price"] == 1299.5 and pd.isna(cat.iloc[2]["brand"])
    assert first["product_text"].startswith("Steel Kettle. Brand: Brewly. Category: Kettles. Features: steel, cordless")


def test_woocommerce_sale_price_first_and_sku_or_id(tmp_path):
    cat, _ = import_feed(read_feed(write(tmp_path, "w.csv", WOO_CSV)), "woocommerce")
    assert cat["product_id"].tolist() == ["K-1", "12"]
    assert cat["price"].tolist() == [19.99, 15.0] and cat["category"].tolist() == ["Kettles", "Coffee"]


def test_shopify_products_json_and_custom_mapping(tmp_path):
    data = {"products": [{"id": 1, "handle": "mug", "title": "Mug", "vendor": "Clay Co", "product_type": "Mugs",
                          "body_html": "<p>350 ml</p>", "tags": ["ceramic", "gift"], "variants": [{"price": "12.00"}]}]}
    cat, _ = import_feed(read_feed(write(tmp_path, "p.json", json.dumps(data))), "shopify_json")
    assert cat.iloc[0][["product_id", "brand", "category", "price", "features"]].tolist() == \
        ["mug", "Clay Co", "Mugs", 12.0, "ceramic; gift"]
    feed = pd.DataFrame({"code": ["a1", "a1"], "Name": ["Lamp", "Lamp copy"], "Cost": ["$10", "abc"]})
    cat, log = import_feed(feed, "generic", {"product_id": "code", "title": "Name", "price": "Cost"})
    assert cat["title"].tolist() == ["Lamp"] and log["dropped_duplicate_id"] == 1
    with pytest.raises(ValueError, match="no column"):
        import_feed(feed, "generic")
    with pytest.raises(ValueError, match="unknown field"):
        import_feed(feed, "generic", {"colour": "x"})


class FakeModel:
    def encode(self, texts, **kwargs):
        rng = np.random.default_rng(0)
        e = rng.normal(size=(len(texts), 4)).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_build_store_round_trips_through_load_resources(tmp_path, monkeypatch):
    from src import embed, pipeline

    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    monkeypatch.setattr(embed, "load_model", lambda name: FakeModel())
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8},
           "llm": {"model": "m", "cache_path": str(tmp_path / "c.jsonl"), "call_log_path": str(tmp_path / "l.jsonl"),
                   "rpm": 10, "temperature": 0, "max_output_tokens": 100}}
    log = store.build_store("demo", write(tmp_path, "s.csv", SHOPIFY_CSV), "shopify", model=FakeModel(), cfg=cfg)
    assert log["products"] == 3 and (tmp_path / "stores" / "demo" / "import_log.json").exists()
    assert store.list_stores() == ["demo"]
    res = pipeline.load_resources(cfg, store="demo")
    assert len(res.catalog) == 3 and res.embeddings.shape == (3, 4) and res.profiles == {}
    assert res.meta.categories == ["Kettles", "Toasters"]
