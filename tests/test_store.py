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


def test_non_finite_rating_counts_are_unknown_not_a_crash():
    feed = pd.DataFrame({"product_id": ["a", "b"], "title": ["A", "B"], "rating": [4.0, 4.0],
                         "rating_count": ["inf", "1e400"]})
    cat, _ = import_feed(feed, "generic")
    assert cat["rating_count"].tolist() == [0, 0]


def test_column_mapping_questions_cover_the_optional_rating_columns():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(__file__).parents[1].joinpath("app", "streamlit_app.py").read_text(encoding="utf-8"))
    keys = next(ast.literal_eval(node.value).keys() for node in tree.body if isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "FIELD_QUESTIONS")
    assert set(store.FIELDS) | set(store.OPTIONAL_FIELDS) <= set(keys)


# ---- Phase 13.2: stock / availability and incremental re-sync -------------------------------------------

def test_shopify_csv_stock_combines_variant_rows_and_selling_policy(tmp_path):
    csv = """Handle,Title,Variant Price,Variant Inventory Tracker,Variant Inventory Qty,Variant Inventory Policy
sold-out,Sold Out Kettle,10,shopify,0,deny
preorder,Preorder Kettle,10,shopify,0,continue
mixed,Mixed Kettle,10,shopify,0,deny
mixed,,12,shopify,3,deny
untracked,Untracked Kettle,10,,0,deny
"""
    cat, log = import_feed(read_feed(write(tmp_path, "s.csv", csv)), "shopify")
    stock = dict(zip(cat["product_id"], cat["in_stock"]))
    assert stock == {"sold-out": False, "preorder": True, "mixed": True, "untracked": True}
    assert log["out_of_stock"] == 1
    plain, _ = import_feed(read_feed(write(tmp_path, "p.csv", SHOPIFY_CSV)), "shopify")   # no inventory columns
    assert plain["in_stock"].isna().all()


def test_shopify_json_woocommerce_and_generic_stock(tmp_path):
    data = {"products": [
        {"id": 1, "handle": "a", "title": "A", "variants": [{"price": "5", "available": False},
                                                             {"price": "6", "available": True}]},
        {"id": 2, "handle": "b", "title": "B", "variants": [{"price": "5", "available": False}]},
        {"id": 3, "handle": "c", "title": "C", "variants": [{"price": "5"}]}]}
    cat, _ = import_feed(read_feed(write(tmp_path, "p.json", json.dumps(data))), "shopify_json")
    assert cat["in_stock"].tolist() == [True, False, None]
    woo = """ID,SKU,Type,Parent,Name,Regular price,In stock?,Backorders allowed?
1,K-1,simple,,Kettle,20,0,0
2,K-2,simple,,Frother,15,0,notify
3,K-3,variable,,Grinder,,,
4,,variation,id:3,Grinder - red,30,1,0
5,K-5,simple,,Scale,9,1,0
"""
    cat, _ = import_feed(read_feed(write(tmp_path, "w.csv", woo)), "woocommerce")
    assert dict(zip(cat["product_id"], cat["in_stock"])) == {"K-1": False, "K-2": True, "K-3": True, "K-5": True}
    feed = pd.DataFrame({"product_id": list("abcde"), "title": list("ABCDE"),
                         "availability": ["in stock", "Out of Stock", "0", "12", ""]})
    cat, _ = import_feed(feed, "generic")
    assert cat["in_stock"].tolist() == [True, False, False, True, None]
    mapped, _ = import_feed(feed.rename(columns={"availability": "Stock status"}), "generic",
                            {"in_stock": "Stock status"})            # ("Status" means published, in Shopify)
    assert mapped["in_stock"].tolist()[:2] == [True, False]


class CountingModel(FakeModel):
    def __init__(self):
        self.encoded = []

    def encode(self, texts, **kwargs):
        self.encoded += list(texts)
        return super().encode(texts, **kwargs)


def test_refresh_reuses_unchanged_embeddings_and_reports_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8}}
    feed = pd.DataFrame({"product_id": ["a", "b", "c"], "title": ["Kettle", "Toaster", "Grinder"],
                         "price": [10, 20, 30], "in_stock": [1, 1, 1]})
    path = tmp_path / "feed.csv"
    feed.to_csv(path, index=False)
    store.build_store("demo", path, "generic", model=CountingModel(), cfg=cfg)
    old_emb = np.load(tmp_path / "stores" / "demo" / "embeddings.npy")
    feed = pd.DataFrame({"product_id": ["a", "b", "d"], "title": ["Kettle", "Toaster XL", "Scale"],
                         "price": [12, 20, 5], "in_stock": [0, 1, 1]})              # a: price + stock only
    feed.to_csv(path, index=False)
    model = CountingModel()
    log = store.refresh_store("demo", model=model, cfg=cfg)
    assert (log["added"], log["removed"], log["changed"], log["reused"]) == (1, 1, 1, 1)
    assert len(model.encoded) == 2                                   # only b (new text) and d (new product)
    cat = pd.read_pickle(tmp_path / "stores" / "demo" / "catalog.pkl")
    assert cat.set_index("product_id").loc["a", "price"] == 12 and not cat.set_index("product_id").loc["a", "in_stock"]
    new_emb = np.load(tmp_path / "stores" / "demo" / "embeddings.npy")
    assert np.array_equal(new_emb[list(cat["product_id"]).index("a")], old_emb[0])
    assert json.loads((tmp_path / "stores" / "demo" / "import_log.json").read_text())["source"] == str(path)


def test_refresh_needs_a_source_it_can_read_again(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8}}
    feed = pd.DataFrame({"product_id": ["a"], "title": ["Kettle"]})
    store.build_store_from_feed("up", feed, "generic", model=FakeModel(), cfg=cfg, source="upload")
    with pytest.raises(ValueError, match="upload"):
        store.refresh_store("up", model=FakeModel(), cfg=cfg)
    with pytest.raises(ValueError, match="no store"):
        store.refresh_store("missing", model=FakeModel(), cfg=cfg)
    calls, shop_feed = [], pd.DataFrame({"handle": ["a"], "title": ["Kettle"]})
    monkeypatch.setattr(store, "read_feed", lambda src: calls.append(src) or shop_feed)
    store.build_store_from_feed("shop", shop_feed, "shopify_json", model=FakeModel(), cfg=cfg, source="MyShop.com")
    store.refresh_store("shop", model=FakeModel(), cfg=cfg)
    assert calls == ["https://myshop.com/products.json"]              # a typed Shopify address is re-read



# ---- verifier findings, Phase 13.2 ----------------------------------------------------------------------

def test_shopify_stock_without_a_tracker_column_respects_quantity_and_policy(tmp_path):
    csv = """Handle,Title,Variant Price,Variant Inventory Qty,Variant Inventory Policy
gone,Gone Kettle,10,0,deny
here,Here Kettle,10,4,deny
"""
    cat, _ = import_feed(read_feed(write(tmp_path, "s.csv", csv)), "shopify")
    assert cat["in_stock"].tolist() == [False, True]


def test_woocommerce_variations_that_name_their_parent_by_sku(tmp_path):
    woo = """ID,SKU,Type,Parent,Name,Regular price,In stock?
3,G-1,variable,,Grinder,,
4,,variation,G-1,Grinder - red,30,0
5,,variation,G-1,Grinder - blue,32,0
"""
    cat, _ = import_feed(read_feed(write(tmp_path, "w.csv", woo)), "woocommerce")
    assert cat["in_stock"].tolist() == [False] and cat["price"].tolist() == [30.0]


def test_odd_stock_words(tmp_path):
    assert store.parse_stock("nan") is None
    assert store.parse_stock("https://schema.org/OutOfStock") is False
    assert store.parse_stock("http://schema.org/InStock") is True
    assert store.parse_stock(-2) is False


def test_refresh_of_a_shopify_address_typed_with_https_reads_products_json(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8}}
    calls, shop_feed = [], pd.DataFrame({"handle": ["a"], "title": ["Kettle"]})
    monkeypatch.setattr(store, "read_feed", lambda src: calls.append(src) or shop_feed)
    store.build_store_from_feed("shop", shop_feed, "shopify_json", model=FakeModel(), cfg=cfg,
                                source="https://MyShop.com/collections/sale?x=1")
    store.refresh_store("shop", model=FakeModel(), cfg=cfg)
    assert calls == ["https://myshop.com/products.json"]


def test_cli_file_sources_are_saved_as_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"product_id": ["a"], "title": ["Kettle"]}).to_csv("feed.csv", index=False)
    log = store.build_store("demo", "feed.csv", "generic", model=FakeModel(),
                            cfg={"seed": 42, "embedding": {"model": "fake", "batch_size": 8}})
    assert log["source"] == str(tmp_path / "feed.csv")


def test_failed_save_keeps_the_previous_store_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8}}
    store.build_store_from_feed("demo", pd.DataFrame({"product_id": ["a", "b"], "title": ["Kettle", "Toaster"]}),
                                "generic", model=FakeModel(), cfg=cfg, source="upload")
    d = tmp_path / "stores" / "demo"
    before = (np.load(d / "embeddings.npy"), np.load(d / "product_ids.npy"), pd.read_pickle(d / "catalog.pkl"))
    real_replace = store.os.replace

    def failing_replace(src, dst):
        if str(dst).endswith("catalog.pkl"):
            raise PermissionError("catalog in use")
        return real_replace(src, dst)

    monkeypatch.setattr(store.os, "replace", failing_replace)
    with pytest.raises(PermissionError):
        store.build_store_from_feed("demo", pd.DataFrame({"product_id": ["c"], "title": ["Scale"]}), "generic",
                                    model=FakeModel(), cfg=cfg, source="upload")
    assert np.array_equal(np.load(d / "embeddings.npy"), before[0])
    assert np.load(d / "product_ids.npy").tolist() == before[1].tolist()
    assert pd.read_pickle(d / "catalog.pkl")["product_id"].tolist() == ["a", "b"]


def test_vectors_are_reused_only_when_the_log_names_the_same_model(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8}}
    feed = pd.DataFrame({"product_id": ["a"], "title": ["Kettle"]})
    store.build_store_from_feed("demo", feed, "generic", model=FakeModel(), cfg=cfg, source="upload")
    log_path = tmp_path / "stores" / "demo" / "import_log.json"
    log = json.loads(log_path.read_text())
    log.pop("embedding_model")                                       # an import from before Phase 13.2
    log_path.write_text(json.dumps(log))
    assert store.previous_vectors(tmp_path / "stores" / "demo", "fake") == ({}, {})
    (tmp_path / "stores" / "demo" / "catalog.pkl").write_bytes(b"not a pickle")
    assert store.previous_vectors(tmp_path / "stores" / "demo", "fake") == ({}, {})


def test_cli_refresh_rejects_negative_interval_and_ignored_options(tmp_path, monkeypatch):
    for argv in (["--name", "x", "--refresh", "--every", "-5"], ["--name", "x", "--refresh", "--file", "f.csv"]):
        with pytest.raises(SystemExit) as exc:
            store.main(argv)
        assert exc.value.code == 2                                       # rejected by the argument parser



# ---- Phase 13.4: webhooks (one-product updates) and per-store secrets ----------------------------------

SHOP_PRODUCT = {"id": 101, "handle": "steel-kettle", "title": "Steel Kettle", "vendor": "Brewly",
                "product_type": "Kettles", "body_html": "<p>Fast boil</p>", "tags": "steel, cordless",
                "status": "active", "variants": [{"price": "39.00", "inventory_management": "shopify",
                                                  "inventory_policy": "deny", "inventory_quantity": 4}]}


def webhook_store(tmp_path, monkeypatch, model=None):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8}}
    feed = pd.DataFrame([store.flatten_shopify({**SHOP_PRODUCT, "variants": [{"price": "39.00"}]}),
                         store.flatten_shopify({"id": 102, "handle": "glass-kettle", "title": "Glass Kettle",
                                                "variants": [{"price": "25.00"}]})])
    store.build_store_from_feed("shop", feed, "shopify_json", model=FakeModel(), cfg=cfg, source="shop.com")
    return cfg


def test_shopify_webhook_updates_one_product_and_reembeds_only_changed_text(tmp_path, monkeypatch):
    cfg = webhook_store(tmp_path, monkeypatch)
    model = CountingModel()
    sold_out = {**SHOP_PRODUCT, "variants": [{**SHOP_PRODUCT["variants"][0], "price": "35.00",
                                               "inventory_quantity": 0}]}
    log = store.apply_store_changes("shop", store.shopify_webhook_feed(sold_out), model=model, cfg=cfg)
    cat = pd.read_pickle(tmp_path / "stores" / "shop" / "catalog.pkl").set_index("product_id")
    assert cat.loc["steel-kettle", "price"] == 35.0 and cat.loc["steel-kettle", "in_stock"] is False
    assert model.encoded == [] and log["products"] == 2 and log["source"] == "shop.com"   # text unchanged
    new = {**SHOP_PRODUCT, "id": 103, "handle": "copper-kettle", "title": "Copper Kettle"}
    store.apply_store_changes("shop", store.shopify_webhook_feed(new), model=model, cfg=cfg)
    assert len(model.encoded) == 1
    draft = {**SHOP_PRODUCT, "status": "draft"}                         # unpublished -> removed
    store.apply_store_changes("shop", store.shopify_webhook_feed(draft), model=model, cfg=cfg)
    store.apply_store_changes("shop", delete_ids=["102"], model=model, cfg=cfg)   # delete payloads carry the id
    cat = pd.read_pickle(tmp_path / "stores" / "shop" / "catalog.pkl")
    assert cat["product_id"].tolist() == ["copper-kettle"]
    emb, ids = np.load(tmp_path / "stores" / "shop" / "embeddings.npy"), np.load(
        tmp_path / "stores" / "shop" / "product_ids.npy")
    assert ids.tolist() == ["copper-kettle"] and len(emb) == 1


def test_woocommerce_webhook_feed_maps_stock_status_and_publishing():
    product = {"id": 7, "sku": "K-7", "name": "Kettle", "type": "simple", "regular_price": "20", "sale_price": "18",
               "description": "Boils", "short_description": "Fast", "categories": [{"name": "Kettles"}],
               "tags": [{"name": "steel"}], "stock_status": "outofstock", "status": "publish"}
    cat, _ = import_feed(store.woocommerce_webhook_feed(product), "woocommerce")
    assert cat[["product_id", "price", "category"]].values.tolist() == [["K-7", 18.0, "Kettles"]]
    assert cat["in_stock"].tolist() == [False] and cat["source_id"].tolist() == ["7"]
    hidden, _ = import_feed(store.woocommerce_webhook_feed({**product, "status": "draft"}), "woocommerce")
    assert hidden.empty


def test_store_secrets_are_created_once_and_kept(tmp_path, monkeypatch):
    webhook_store(tmp_path, monkeypatch)
    first = store.store_secrets("shop")
    assert len(first["webhook_secret"]) >= 32 and len(first["api_key"]) >= 32
    assert store.store_secrets("shop") == first
    with pytest.raises(ValueError, match="no store"):
        store.store_secrets("missing")



def test_webhook_secret_can_be_set_from_the_platform(tmp_path, monkeypatch):
    webhook_store(tmp_path, monkeypatch)
    store.set_webhook_secret("shop", "shpss_" + "a" * 26)
    assert store.store_secrets("shop")["webhook_secret"] == "shpss_" + "a" * 26
    with pytest.raises(ValueError, match="16 characters"):
        store.set_webhook_secret("shop", "short")


def test_webhooks_work_for_a_store_imported_from_a_shopify_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8}}
    store.build_store("csv", write(tmp_path, "s.csv", SHOPIFY_CSV), "shopify", model=FakeModel(), cfg=cfg)
    product = {**SHOP_PRODUCT, "variants": [{"price": "30.00", "inventory_management": "shopify",
                                             "inventory_policy": "deny", "inventory_quantity": 0}]}
    store.apply_store_changes("csv", store.shopify_webhook_feed(product), preset="shopify_json", model=FakeModel(),
                              cfg=cfg)
    cat = pd.read_pickle(tmp_path / "stores" / "csv" / "catalog.pkl").set_index("product_id")
    assert cat.loc["steel-kettle", "price"] == 30.0 and cat.loc["steel-kettle", "in_stock"] is False
    assert len(cat) == 3                                             # same product id (the handle), not a new one



# ---- verifier findings, Phase 13.4 ----------------------------------------------------------------------

def csv_store(tmp_path, monkeypatch, site="shop.com"):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8}}
    feed = read_feed(write(tmp_path, "s.csv", SHOPIFY_CSV))
    store.build_store_from_feed("csv", feed, "shopify", model=FakeModel(), cfg=cfg, source="upload", site=site)
    return cfg


def test_shopify_delete_by_id_on_a_csv_store_checks_the_live_store(tmp_path, monkeypatch):
    cfg = csv_store(tmp_path, monkeypatch)
    live = {"products": [{"id": 1, "handle": "steel-kettle", "title": "Steel Kettle"},
                         {"id": 3, "handle": "toaster-2", "title": "2-Slice Toaster"}]}     # glass-kettle deleted
    monkeypatch.setattr(store, "fetch_products_json", lambda url: live)
    log = store.apply_store_changes("csv", delete_ids=["2"], model=FakeModel(), cfg=cfg)
    cat = pd.read_pickle(tmp_path / "stores" / "csv" / "catalog.pkl")
    assert sorted(cat["product_id"]) == ["steel-kettle", "toaster-2"] and log["removed"] == 1
    assert dict(zip(cat["product_id"], cat["source_id"])) == {"steel-kettle": "1", "toaster-2": "3"}


def test_shopify_delete_of_an_unknown_id_without_a_store_address_reports_not_found(tmp_path, monkeypatch):
    cfg = csv_store(tmp_path, monkeypatch, site=None)
    log = store.apply_store_changes("csv", delete_ids=["999"], model=FakeModel(), cfg=cfg)
    assert log["not_found"] == ["999"] and len(pd.read_pickle(tmp_path / "stores" / "csv" / "catalog.pkl")) == 3


def test_renamed_handle_replaces_the_old_row(tmp_path, monkeypatch):
    cfg = webhook_store(tmp_path, monkeypatch)
    store.apply_store_changes("shop", store.shopify_webhook_feed(SHOP_PRODUCT), preset="shopify_json",
                              model=FakeModel(), cfg=cfg)                   # learns source id 101
    store.apply_store_changes("shop", store.shopify_webhook_feed({**SHOP_PRODUCT, "handle": "kettle-renamed"}),
                              preset="shopify_json", model=FakeModel(), cfg=cfg)
    cat = pd.read_pickle(tmp_path / "stores" / "shop" / "catalog.pkl")
    assert sorted(cat["product_id"]) == ["glass-kettle", "kettle-renamed"]


def test_webhook_keeps_brand_category_and_ratings_it_does_not_carry(tmp_path, monkeypatch):
    cfg = csv_store(tmp_path, monkeypatch)                                  # category from "Product Category"
    store.apply_store_changes("csv", store.shopify_webhook_feed({**SHOP_PRODUCT, "product_type": "Misc"}),
                              preset="shopify_json", model=FakeModel(), cfg=cfg)
    cat = pd.read_pickle(tmp_path / "stores" / "csv" / "catalog.pkl").set_index("product_id")
    assert cat.loc["steel-kettle", "category"] == "Kettles"                   # field sources differ: kept
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "w")
    woo = pd.DataFrame({"product_id": ["K-7"], "title": ["Kettle"], "brand": ["MugCo"], "rating": [4.5],
                        "rating_count": [10]})
    store.build_store_from_feed("woo", woo, "generic", model=FakeModel(), cfg=cfg, source="upload")
    payload = {"id": 7, "sku": "K-7", "name": "Kettle", "status": "publish", "regular_price": "20",
               "stock_status": "instock"}
    store.apply_store_changes("woo", store.woocommerce_webhook_feed(payload), preset="woocommerce",
                              model=FakeModel(), cfg=cfg)
    row = pd.read_pickle(tmp_path / "w" / "woo" / "catalog.pkl").set_index("product_id").loc["K-7"]
    assert row["brand"] == "MugCo" and row["avg_rating"] == 4.5 and row["price"] == 20.0
    branded = store.woocommerce_webhook_feed({**payload, "brands": [{"name": "Brewly"}], "average_rating": "4.0",
                                              "rating_count": 3})
    cat, _ = import_feed(branded, "woocommerce")
    assert cat[["brand", "avg_rating", "rating_count"]].values.tolist() == [["Brewly", 4.0, 3]]


def test_saved_mapping_is_not_applied_to_webhook_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8}}
    woo = read_feed(write(tmp_path, "w.csv", WOO_CSV)).rename(columns={"Name": "Product name"})
    store.build_store_from_feed("woo", woo, "woocommerce", {"title": "Product name"}, model=FakeModel(), cfg=cfg)
    payload = {"id": 11, "sku": "K-1", "name": "Travel Kettle 2", "status": "publish", "regular_price": "26"}
    store.apply_store_changes("woo", store.woocommerce_webhook_feed(payload), preset="woocommerce",
                              model=FakeModel(), cfg=cfg)
    cat = pd.read_pickle(tmp_path / "stores" / "woo" / "catalog.pkl").set_index("product_id")
    assert cat.loc["K-1", "title"] == "Travel Kettle 2"


def test_shopify_webhook_tags_unknown_inventory_and_unpublished_storefront():
    feed = store.shopify_webhook_feed({**SHOP_PRODUCT, "variants": [{"price": "39.00"}]})
    assert feed["tags"].iloc[0] == "steel; cordless" and feed["available"].iloc[0] is None
    hidden = store.shopify_webhook_feed({**SHOP_PRODUCT, "published_at": None})
    assert import_feed(hidden, "shopify_json")[0].empty


def test_store_lock_is_shared_between_processes_and_temp_files_are_private(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    (tmp_path / "stores" / "x").mkdir(parents=True)
    with store.store_lock("x"):
        assert (tmp_path / "stores" / "x" / ".lock").exists()
        with pytest.raises(TimeoutError):
            with store.store_lock("x", timeout=0.2):
                pass
    assert not (tmp_path / "stores" / "x" / ".lock").exists()


def test_read_secrets_never_creates_them(tmp_path, monkeypatch):
    webhook_store(tmp_path, monkeypatch)
    assert store.read_secrets("shop") is None and store.read_secrets("missing") is None
    created = store.store_secrets("shop")
    assert store.read_secrets("shop") == created


# ---- replay / ordering guard (signed updated_at) ----------------------------------------------------------

def test_webhook_updates_older_than_the_stored_version_are_skipped(tmp_path, monkeypatch):
    cfg = webhook_store(tmp_path, monkeypatch)

    def send(price, when):
        product = {**SHOP_PRODUCT, "updated_at": when, "variants": [{**SHOP_PRODUCT["variants"][0], "price": price}]}
        return store.apply_store_changes("shop", store.shopify_webhook_feed(product), preset="shopify_json",
                                         model=FakeModel(), cfg=cfg)

    def price():
        return pd.read_pickle(tmp_path / "stores" / "shop" / "catalog.pkl").set_index("product_id").loc[
            "steel-kettle", "price"]

    assert send("30.00", "2026-10-30T10:00:00-04:00")["applied"] == 1 and price() == 30.0
    old = send("20.00", "2026-10-30T09:00:00-04:00")                      # older: out of order or replayed
    assert old["applied"] == 0 and old["stale"] == ["steel-kettle"] and price() == 30.0
    assert send("25.00", "2026-10-30T13:59:59Z")["stale"] == ["steel-kettle"]   # a second earlier, other zone
    assert send("35.00", "2026-10-30T14:00:01Z")["applied"] == 1 and price() == 35.0
    woo = store.woocommerce_webhook_feed({"id": 7, "sku": "K-7", "name": "Kettle", "status": "publish",
                                          "date_modified_gmt": "2026-10-30T12:00:00"})
    assert woo["date_modified_gmt"].iloc[0] == "2026-10-30T12:00:00"



# ---- verifier findings: replay guard (13.4) ------------------------------------------------------------

def guarded(tmp_path, monkeypatch):
    cfg = webhook_store(tmp_path, monkeypatch)

    def send(when, price="30.00", **extra):
        product = {**SHOP_PRODUCT, "updated_at": when, **extra,
                   "variants": [{**SHOP_PRODUCT["variants"][0], "price": price}]}
        return store.apply_store_changes("shop", store.shopify_webhook_feed(product), preset="shopify_json",
                                         model=FakeModel(), cfg=cfg)

    def catalog():
        return pd.read_pickle(tmp_path / "stores" / "shop" / "catalog.pkl").set_index("product_id")

    return cfg, send, catalog


def test_older_unpublish_does_not_remove_a_newer_live_product(tmp_path, monkeypatch):
    cfg, send, catalog = guarded(tmp_path, monkeypatch)
    send("2026-10-30T12:00:00Z")
    send("2026-10-30T11:00:00Z", status="draft")                          # late retry of an older draft
    assert "steel-kettle" in catalog().index


def test_deleted_product_is_not_brought_back_by_an_older_update(tmp_path, monkeypatch):
    cfg, send, catalog = guarded(tmp_path, monkeypatch)
    send("2026-10-30T12:00:00Z")
    store.apply_store_changes("shop", delete_ids=["101"], model=FakeModel(), cfg=cfg)
    log = send("2026-10-30T12:00:00Z", price="31.00")                     # retried update from before the delete
    assert "steel-kettle" not in catalog().index and log["stale"] == ["steel-kettle"]
    send("2099-01-01T00:00:00Z", price="33.00")                            # edited after the delete: restored
    assert catalog().loc["steel-kettle", "price"] == 33.0


def test_second_edit_in_the_same_second_is_applied(tmp_path, monkeypatch):
    cfg, send, catalog = guarded(tmp_path, monkeypatch)
    send("2026-10-30T12:00:00Z", price="30.00")
    send("2026-10-30T12:00:00Z", price="31.00")
    assert catalog().loc["steel-kettle", "price"] == 31.0


def test_payload_without_a_time_keeps_the_stored_time(tmp_path, monkeypatch):
    cfg, send, catalog = guarded(tmp_path, monkeypatch)
    send("2026-10-30T12:00:00Z", price="30.00")
    product = {**SHOP_PRODUCT, "variants": [{**SHOP_PRODUCT["variants"][0], "price": "50.00"}]}
    store.apply_store_changes("shop", store.shopify_webhook_feed(product), preset="shopify_json",
                              model=FakeModel(), cfg=cfg)                 # no updated_at: applied, time kept
    assert catalog().loc["steel-kettle", "price"] == 50.0
    send("2026-10-30T11:00:00Z", price="1.00")
    assert catalog().loc["steel-kettle", "price"] == 50.0


def test_delete_after_a_stale_update_is_not_reported_as_stale_and_noop_updates_do_not_rewrite(tmp_path, monkeypatch):
    cfg, send, catalog = guarded(tmp_path, monkeypatch)
    send("2026-10-30T12:00:00Z")
    before = (tmp_path / "stores" / "shop" / "catalog.pkl").stat().st_mtime_ns
    assert send("2026-10-30T11:00:00Z")["stale"] == ["steel-kettle"]
    assert (tmp_path / "stores" / "shop" / "catalog.pkl").stat().st_mtime_ns == before   # nothing written
    log = store.apply_store_changes("shop", delete_ids=["102"], model=FakeModel(), cfg=cfg)
    assert log["stale"] == [] and "glass-kettle" not in catalog().index


def test_refresh_keeps_a_product_a_newer_webhook_updated(tmp_path, monkeypatch):
    cfg, send, catalog = guarded(tmp_path, monkeypatch)
    send("2026-10-30T12:00:00Z", price="30.00")
    old_snapshot = pd.DataFrame([store.flatten_shopify({**SHOP_PRODUCT, "updated_at": "2026-10-30T10:00:00Z",
                                                        "variants": [{"price": "20.00"}]}),
                                 store.flatten_shopify({"id": 102, "handle": "glass-kettle", "title": "Glass Kettle",
                                                        "variants": [{"price": "26.00"}]})])
    store.build_store_from_feed("shop", old_snapshot, "shopify_json", model=FakeModel(), cfg=cfg, source="shop.com")
    assert catalog().loc["steel-kettle", "price"] == 30.0 and catalog().loc["glass-kettle", "price"] == 26.0


def test_products_without_platform_ids_do_not_share_timestamps():
    cat = pd.DataFrame({"product_id": ["a", "b"], "source_id": [None, None],
                        "source_updated_at": pd.to_datetime([None, "2026-10-30T12:00:00Z"], utc=True)})
    new = pd.DataFrame({"product_id": ["c"], "source_id": [None],
                        "source_updated_at": pd.to_datetime(["2026-10-30T11:00:00Z"], utc=True)})
    kept, stale = store.drop_stale(new, cat)
    assert stale == [] and kept["product_id"].tolist() == ["c"]



# ---- Phase 13.8 (E): shopper events feed popularity -----------------------------------------------------

def events_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    cfg = {"seed": 42, "embedding": {"model": "fake", "batch_size": 8}}
    feed = pd.DataFrame({"product_id": ["a", "b", "c"], "title": ["Kettle", "Toaster", "Grinder"]})
    store.build_store_from_feed("shop", feed, "generic", model=FakeModel(), cfg=cfg, source="upload")
    return cfg


def test_record_events_keeps_only_known_products_and_types(tmp_path, monkeypatch):
    events_store(tmp_path, monkeypatch)
    kept = store.record_events("shop", "client-1", [
        {"type": "click", "product_id": "a"}, {"type": "buy", "product_id": "zzz"},
        {"type": "hack", "product_id": "a"}, {"type": "save", "product_id": "b", "query": "secret"}] + [
        {"type": "shown", "product_id": "c"}] * 60)
    lines = [json.loads(x) for x in (tmp_path / "stores" / "shop" / "events.jsonl").read_text().splitlines()]
    assert kept == len(lines) == 50                                   # at most 50 per request
    assert {tuple(sorted(e)) for e in lines} == {("client", "product_id", "ts", "type")}   # nothing else stored
    assert "secret" not in (tmp_path / "stores" / "shop" / "events.jsonl").read_text()
    with pytest.raises(ValueError, match="no store"):
        store.record_events("missing", "c", [{"type": "click", "product_id": "a"}])


def test_event_popularity_weights_and_counts_each_browser_once_a_day(tmp_path, monkeypatch):
    events_store(tmp_path, monkeypatch)
    for client in ("x", "y"):
        store.record_events("shop", client, [{"type": "buy", "product_id": "a"}, {"type": "click", "product_id": "a"}])
    store.record_events("shop", "x", [{"type": "save", "product_id": "b"}] * 5)     # repeats count once
    store.record_events("shop", "x", [{"type": "shown", "product_id": "c"}])
    pop = store.event_popularity("shop")
    assert pop.to_dict() == {"a": 12, "b": 3}                         # 2 x (buy 5 + click 1); one save; shown 0


def test_update_popularity_reembeds_nothing_and_survives_refresh(tmp_path, monkeypatch):
    cfg = events_store(tmp_path, monkeypatch)
    store.record_events("shop", "x", [{"type": "buy", "product_id": "b"}])
    model = CountingModel()
    store.update_popularity("shop", model=model, cfg=cfg)
    cat = pd.read_pickle(tmp_path / "stores" / "shop" / "catalog.pkl").set_index("product_id")
    assert cat.loc["b", "popularity"] == 5 and cat.loc["a", "popularity"] == 0 and model.encoded == []
    feed = pd.DataFrame({"product_id": ["a", "b"], "title": ["Kettle", "Toaster"]})
    store.build_store_from_feed("shop", feed, "generic", model=model, cfg=cfg, source="upload")   # re-import
    assert pd.read_pickle(tmp_path / "stores" / "shop" / "catalog.pkl").set_index("product_id").loc["b", "popularity"] == 5


def test_event_summary_rates_and_top_products(tmp_path, monkeypatch):
    events_store(tmp_path, monkeypatch)
    store.record_events("shop", "x", [{"type": "shown", "product_id": p} for p in "abc"] +
                        [{"type": "click", "product_id": "a"}, {"type": "buy", "product_id": "a"}])
    store.record_events("shop", "y", [{"type": "shown", "product_id": "a"}, {"type": "save", "product_id": "b"}])
    s = store.event_summary("shop")
    assert s["counts"] == {"shown": 4, "click": 1, "save": 1, "buy": 1} and s["shoppers"] == 2
    assert s["click_rate"] == 0.25 and s["buy_click_rate"] == 0.25
    assert s["top_products"][0] == {"product_id": "a", "score": 6}



# ---- verifier findings, Phase 13.8 ----------------------------------------------------------------------

def test_events_file_is_compacted_and_bounded(tmp_path, monkeypatch):
    events_store(tmp_path, monkeypatch)
    path = tmp_path / "stores" / "shop" / "events.jsonl"
    old = {"ts": "2020-01-01T00:00:00", "client": "z", "type": "buy", "product_id": "a"}
    path.write_text(json.dumps(old) + "\n")
    for _ in range(30):                                                # the same action repeated all day
        store.record_events("shop", "x", [{"type": "click", "product_id": "a"}])
    store.compact_events("shop")
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines == [{**lines[0]}] and lines[0]["client"] == "x"       # old dropped, repeats collapsed
    monkeypatch.setattr(store, "MAX_EVENT_BYTES", 2000)
    for i in range(100):                                               # many browsers: file stays bounded
        store.record_events("shop", f"bot{i}", [{"type": "click", "product_id": "b"}])
    assert path.stat().st_size <= 2000 + 500


def test_event_popularity_is_cached_until_the_file_changes(tmp_path, monkeypatch):
    events_store(tmp_path, monkeypatch)
    store.record_events("shop", "x", [{"type": "buy", "product_id": "a"}])
    reads = []
    real = store._read_events
    monkeypatch.setattr(store, "_read_events", lambda *a, **k: reads.append(1) or real(*a, **k))
    store.event_popularity("shop")
    store.event_popularity("shop")
    assert len(reads) == 1
    store.record_events("shop", "y", [{"type": "buy", "product_id": "a"}])
    assert store.event_popularity("shop").to_dict() == {"a": 10} and len(reads) == 2


def test_malformed_events_are_ignored(tmp_path, monkeypatch):
    events_store(tmp_path, monkeypatch)
    assert store.record_events("shop", "x", [{"type": ["buy"], "product_id": "a"},
                                             {"type": "buy", "product_id": ["a"]}, "junk"]) == 0


def test_summary_needs_a_separate_owner_key(tmp_path, monkeypatch):
    events_store(tmp_path, monkeypatch)
    keys = store.store_secrets("shop")
    assert len(keys["admin_key"]) >= 32 and keys["admin_key"] != keys["api_key"]
    (tmp_path / "stores" / "shop" / "secrets.json").write_text(json.dumps(
        {"webhook_secret": "w" * 40, "api_key": "k" * 40}))           # created before admin keys existed
    assert len(store.store_secrets("shop")["admin_key"]) >= 32           # added, others kept
    assert store.store_secrets("shop")["api_key"] == "k" * 40



# ---- verifier findings, final Phase 13 batch --------------------------------------------------------------

def test_compaction_leaves_room_and_drops_shown_events_first(tmp_path, monkeypatch):
    events_store(tmp_path, monkeypatch)
    monkeypatch.setattr(store, "MAX_EVENT_BYTES", 4000)
    store.record_events("shop", "buyer", [{"type": "buy", "product_id": "a"}])          # oldest line is a buy
    runs = []
    real = store._compact
    monkeypatch.setattr(store, "_compact", lambda name: runs.append(1) or real(name))
    for i in range(200):
        store.record_events("shop", f"v{i}", [{"type": "shown", "product_id": "b"}])
    assert len(runs) <= 10                                             # not on every write
    assert store.event_popularity("shop").get("a") == 5                # the buy survives; shown lines go first


def test_compaction_failure_does_not_break_recording(tmp_path, monkeypatch):
    events_store(tmp_path, monkeypatch)
    monkeypatch.setattr(store, "MAX_EVENT_BYTES", 300)
    monkeypatch.setattr(store.os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError("in use")))
    for i in range(10):
        assert store.record_events("shop", f"v{i}", [{"type": "click", "product_id": "a"}]) == 1
    assert not list((tmp_path / "stores" / "shop").glob("*.tmp"))       # no temp file left behind


def test_import_compacts_old_events(tmp_path, monkeypatch):
    cfg = events_store(tmp_path, monkeypatch)
    path = tmp_path / "stores" / "shop" / "events.jsonl"
    path.write_text(json.dumps({"ts": "2020-01-01T00:00:00", "client": "z", "type": "buy", "product_id": "a"}) + "\n")
    feed = pd.DataFrame({"product_id": ["a"], "title": ["Kettle"]})
    store.build_store_from_feed("shop", feed, "generic", model=FakeModel(), cfg=cfg, source="upload")
    assert path.read_text() == ""                                      # deleted, not just ignored
