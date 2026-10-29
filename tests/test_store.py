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


# ---- verifier findings, Phase 11 (29 Oct): failing tests first -------------------------------------------

def test_store_names_cannot_escape_the_stores_folder(tmp_path, monkeypatch):
    from src import pipeline

    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    for bad in ("../x", "..", "C:/x", "a/b", "a\b", ""):
        assert not store.valid_store_name(bad)
        with pytest.raises(ValueError, match="store name"):
            store.build_store(bad, write(tmp_path, "s.csv", SHOPIFY_CSV), "shopify", model=FakeModel(), cfg={})
        with pytest.raises(ValueError, match="store name"):
            pipeline.load_resources({"seed": 42}, store=bad)
    assert store.valid_store_name("my-store_2")


def test_shopify_taxonomy_level_names_may_contain_commas(tmp_path):
    feed = pd.DataFrame({"Handle": ["beans"], "Title": ["Coffee Beans"],
                         "Product Category": ["Food, Beverages & Tobacco > Beverages > Coffee, Tea & Cocoa"]})
    cat, _ = import_feed(feed, "shopify")
    assert cat.iloc[0]["category"] == "Coffee, Tea & Cocoa"
    woo, _ = import_feed(read_feed(write(tmp_path, "w.csv", WOO_CSV)), "woocommerce")
    assert woo.iloc[0]["category"] == "Kettles"                               # Woo lists categories with commas


def test_unpublished_products_and_woocommerce_variations(tmp_path):
    shop = pd.DataFrame({"Handle": ["a", "b", "b"], "Title": ["Live", "Draft", ""], "Status": ["active", "draft", ""],
                         "Variant Price": ["5", "6", "7"]})
    cat, log = import_feed(shop, "shopify")
    assert cat["product_id"].tolist() == ["a"] and log["dropped_unpublished"] == 1
    woo = pd.DataFrame({"ID": ["11", "12", "13", "14"], "SKU": ["", "", "", ""], "Type": ["variable", "variation",
                        "variation", "simple"], "Parent": ["", "id:11", "id:11", ""], "Published": ["1", "1", "1", "0"],
                        "Name": ["Hoodie", "Hoodie - Blue", "Hoodie - Red", "Hidden"], "Regular price": ["", "30", "25", "9"],
                        "Categories": ["Clothing > Hoodies", "", "", "Clothing"]})
    cat, log = import_feed(woo, "woocommerce")
    assert cat["product_id"].tolist() == ["11"] and cat.iloc[0]["price"] == 25.0   # cheapest variation
    assert cat.iloc[0]["category"] == "Hoodies" and log["dropped_variations"] == 2 and log["dropped_unpublished"] == 1


def test_shopify_products_json_url_reads_every_page(monkeypatch):
    import httpx

    pages = {1: [{"handle": f"p{i}", "title": f"P{i}", "variants": [{"price": "1"}]} for i in range(250)],
             2: [{"handle": "last", "title": "Last", "variants": [{"price": "2"}]}]}

    class Resp:
        def __init__(self, products): self.products = products
        def raise_for_status(self): return self
        def json(self): return {"products": self.products}

    def fake_get(url, **kwargs):
        page = int(url.split("page=")[1].split("&")[0])
        return Resp(pages.get(page, []))

    monkeypatch.setattr(httpx, "get", fake_get)
    feed = read_feed("https://shop.example/products.json")
    assert len(feed) == 251 and feed["handle"].iloc[-1] == "last"


def test_numeric_ids_with_missing_values_stay_integers():
    feed = pd.DataFrame({"product_id": [1, None, 3], "title": ["A", "B", "C"]})
    cat, _ = import_feed(feed, "generic")
    assert cat["product_id"].tolist() == ["1", "3"]


def test_format_detection_and_friendly_names(tmp_path):
    from src.store import detect_preset, shopify_feed_url, store_name_from

    assert detect_preset(read_feed(write(tmp_path, "s.csv", SHOPIFY_CSV))) == "shopify"
    assert detect_preset(read_feed(write(tmp_path, "w.csv", WOO_CSV))) == "woocommerce"
    assert detect_preset(pd.DataFrame(columns=["handle", "title", "vendor"])) == "shopify_json"
    assert detect_preset(pd.DataFrame(columns=["product_id", "title"])) == "generic"
    assert detect_preset(pd.DataFrame(columns=["Item", "Cost"])) is None
    assert store_name_from("My Shop!") == "my-shop" and store_name_from("https://Shop.example.com/x") == "shop-example-com"
    assert store.valid_store_name(store_name_from("../../etc")) and store_name_from("!!!") == "mystore"
    assert shopify_feed_url("https://mystore.com/collections/all") == "https://mystore.com/products.json"
    with pytest.raises(ValueError):
        shopify_feed_url("  ")


def test_shopify_address_is_cleaned_and_local_addresses_refused():
    # Verifier 11b: '?'/'#' leaked into the URL (then fetched as CSV) and local hosts were accepted.
    from src.store import shopify_feed_url

    assert shopify_feed_url("mystore.com?x=1#top") == "https://mystore.com/products.json"
    assert shopify_feed_url("Shop.Example.com:443/") == "https://shop.example.com/products.json"
    for bad in ("localhost:8000", "127.0.0.1", "169.254.169.254", "intranet", "[::1]"):
        with pytest.raises(ValueError):
            shopify_feed_url(bad)


def test_optional_rating_columns_are_read_and_checked():
    feed = pd.DataFrame({"product_id": ["a", "b", "c", "d"], "title": ["A", "B", "C", "D"],
                         "rating": [4.6, "7", "abc", None], "review_count": [12, 3, "x", None]})
    cat, _ = import_feed(feed, "generic")
    assert cat["avg_rating"].tolist()[:1] == [4.6] and cat["avg_rating"].iloc[1:].isna().all()   # 1-5 only
    assert cat["rating_count"].tolist() == [12, 3, 0, 0]
    mapped, _ = import_feed(feed.rename(columns={"rating": "Stars"}), "generic", {"rating": "Stars"})
    assert mapped["avg_rating"].iloc[0] == 4.6
    plain, _ = import_feed(feed[["product_id", "title"]], "generic")
    assert plain["avg_rating"].isna().all() and (plain["rating_count"] == 0).all()
