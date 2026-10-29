"""Store connector: import any store's product export into the canonical schema and build its search index.

Presets map common export formats to the spec 5.1 fields; `mapping` overrides or adds columns for anything else.
Read-only: the store's own data is never modified. Output: stores/<name>/ (catalog.pkl, embeddings, import_log.json).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import ROOT, load_config
from src.data import build_product_text, clean_text, parse_price

STORES_DIR = ROOT / "stores"
STORE_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
PAGE_SIZE, MAX_PAGES = 250, 200     # Shopify products.json pagination
FIELDS = ["product_id", "title", "brand", "category", "price", "description", "features"]
# optional rating columns (generic exports, review apps); mapping keys "rating" and "rating_count" also work
OPTIONAL_FIELDS = {"rating": ["rating", "average_rating", "avg_rating", "Average rating"],
                   "rating_count": ["rating_count", "review_count", "reviews", "Review count"],
                   "in_stock": ["in_stock", "availability", "available", "stock", "In stock?", "Stock"]}
IN_STOCK_WORDS = {"1", "true", "yes", "y", "in stock", "instock", "in_stock", "available", "onbackorder", "backorder"}
OUT_OF_STOCK_WORDS = {"0", "false", "no", "n", "out of stock", "outofstock", "out_of_stock", "sold out", "soldout",
                      "unavailable"}

# canonical field -> source columns, first non-empty one wins
PRESETS: dict[str, dict[str, list[str]]] = {
    "generic": {f: [f] for f in FIELDS},
    "shopify": {                                   # Shopify admin "Export products" CSV
        "product_id": ["Handle"], "title": ["Title"], "brand": ["Vendor"],
        "category": ["Product Category", "Type"], "price": ["Variant Price"],
        "description": ["Body (HTML)"], "features": ["Tags"],
    },
    "shopify_json": {                              # https://<store>/products.json (flattened below)
        "product_id": ["handle", "id"], "title": ["title"], "brand": ["vendor"], "category": ["product_type"],
        "price": ["price"], "description": ["body_html"], "features": ["tags"],
    },
    "woocommerce": {                               # WooCommerce "Export products" CSV
        "product_id": ["SKU", "ID"], "title": ["Name"], "brand": ["Brands"], "category": ["Categories"],
        "price": ["Sale price", "Regular price"], "description": ["Description"],
        "features": ["Short description", "Tags"],
    },
}


def valid_store_name(name: str) -> bool:
    """Store names are plain identifiers, so a name can never point outside stores/."""
    return isinstance(name, str) and STORE_NAME_RE.fullmatch(name) is not None


def check_store_name(name: str) -> None:
    if not valid_store_name(name):
        raise ValueError(f"invalid store name {name!r}: use 1-64 letters, digits, '-' or '_'")


def _get_json(url: str) -> Any:
    import httpx

    return httpx.get(url, timeout=30, follow_redirects=True).raise_for_status().json()


def fetch_products_json(url: str) -> Any:
    """GET a JSON feed; a Shopify products.json URL is read page by page (Shopify returns at most 250 per page)."""
    base = url + ("&" if "?" in url else "?")
    data = _get_json(f"{base}limit={PAGE_SIZE}&page=1")
    if not (isinstance(data, dict) and "products" in data):
        return data
    products, page, batch = list(data["products"]), 1, data["products"]
    while len(batch) >= PAGE_SIZE and page < MAX_PAGES:
        page += 1
        batch = _get_json(f"{base}limit={PAGE_SIZE}&page={page}").get("products", [])
        products += batch
    return {"products": products}


def read_feed(source: str | Path) -> pd.DataFrame:
    """Read a CSV or JSON export from a path or URL. Shopify products.json is flattened to one row per product."""
    source = str(source)
    if source.lower().split("?")[0].endswith(".json"):
        if source.startswith(("http://", "https://")):
            data: Any = fetch_products_json(source)
        else:
            data = json.loads(Path(source).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "products" in data:
            data = [flatten_shopify(p) for p in data["products"]]
        return pd.DataFrame(data)
    return pd.read_csv(source, dtype=str, keep_default_na=False)


def flatten_shopify(product: dict) -> dict:
    """One Shopify API product -> flat row; price = first variant's price, tags joined with semicolons."""
    variants = product.get("variants") or [{}]
    tags = product.get("tags") or []
    known = [bool(v["available"]) for v in variants if v.get("available") is not None]
    return {**{k: product.get(k) for k in ("id", "handle", "title", "vendor", "product_type", "body_html")},
            "price": variants[0].get("price"), "tags": "; ".join(tags) if isinstance(tags, list) else tags,
            "available": any(known) if known else None}       # in stock when any variant can be bought


def last_level(category: str, multi: bool = False) -> str:
    """Deepest level of a category path. multi=True (WooCommerce lists several categories with commas):
    'Kitchen > Kettles, Gifts' -> 'Kettles'. Otherwise commas belong to level names ('Coffee, Tea & Cocoa')."""
    path = str(category).split(",")[0] if multi else str(category)
    return clean_text(path.split(">")[-1])


def _text(feed: pd.DataFrame, column: str) -> pd.Series:
    return feed[column].astype(str).str.strip() if column in feed else pd.Series("", index=feed.index)


def parse_stock(value: Any) -> bool | None:
    """In stock (True), out of stock (False) or unknown (None) from a stock flag, word or quantity."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None if value is None else bool(value)
    if isinstance(value, (int, float, np.number)):
        return None if np.isnan(value) else bool(value > 0)              # a quantity
    text = str(value).strip().lower()
    if text in ("", "nan", "none", "null", "n/a"):
        return None
    if text.startswith(("http://", "https://")):                          # schema.org/InStock style values
        text = text.rstrip("/").rsplit("/", 1)[-1]
    if text in IN_STOCK_WORDS:
        return True
    if text in OUT_OF_STOCK_WORDS:
        return False
    try:
        return float(text.replace(",", "")) > 0                 # a quantity
    except ValueError:
        return None


def row_stock(feed: pd.DataFrame, preset: str, column: str | None = None) -> pd.Series:
    """Stock per feed row. Shopify CSV: quantity > 0, or "continue selling", or inventory not tracked;
    WooCommerce: "In stock?" or backorders allowed; otherwise the first stock-like column (or `column`)."""
    unknown = pd.Series([None] * len(feed), index=feed.index, dtype=object)
    if column:
        return feed[column].map(parse_stock).astype(object) if column in feed else unknown
    if preset == "shopify" and "Variant Inventory Qty" in feed:
        qty = pd.to_numeric(_text(feed, "Variant Inventory Qty").str.replace(",", ""), errors="coerce")
        tracked = _text(feed, "Variant Inventory Tracker").ne("") if "Variant Inventory Tracker" in feed \
            else pd.Series(True, index=feed.index)                      # no tracker column: quantities count
        sellable = (qty > 0) | _text(feed, "Variant Inventory Policy").str.lower().eq("continue") | ~tracked
        return sellable.astype(object).where(qty.notna(), None)          # rows without a variant say nothing
    if preset == "woocommerce" and "In stock?" in feed:
        backorders = _text(feed, "Backorders allowed?").str.lower().isin(["1", "notify"])
        return feed["In stock?"].map(parse_stock).astype(object).where(~backorders, True)
    for name in OPTIONAL_FIELDS["in_stock"]:
        if name in feed:
            return feed[name].map(parse_stock).astype(object)
    return unknown


def _any_stock(values: pd.Series) -> bool | None:
    known = [bool(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return any(known) if known else None


def drop_unpublished(feed: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Shopify Status other than active, WooCommerce Published other than 1 (blank = not stated, kept)."""
    status = _text(feed, "Status").str.lower()
    published = _text(feed, "Published")
    hidden = (status.ne("") & status.ne("active")) | (published.ne("") & published.ne("1"))
    log["dropped_unpublished"] = int(hidden.sum())
    return feed[~hidden]


def merge_woo_variations(feed: pd.DataFrame, log: dict) -> pd.DataFrame:
    """WooCommerce exports a variable product as a parent row plus one row per variation: keep the parent and
    give it the cheapest variation price when it has none."""
    is_var = _text(feed, "Type").str.contains("variation")
    log["dropped_variations"] = int(is_var.sum())
    if not is_var.any():
        return feed
    var = feed[is_var]
    price = _text(var, "Sale price").where(_text(var, "Sale price").ne(""), _text(var, "Regular price"))
    cheapest = pd.Series(price.map(parse_price).to_numpy(), index=_text(var, "Parent").str.removeprefix("id:"))         .groupby(level=0).min()
    out = feed[~is_var].copy()
    for column in ("Sale price", "Regular price"):
        if column not in out:
            out[column] = ""
    if "In stock?" in feed:                    # a variable parent is in stock when any variation is
        var_stock = pd.Series(var["In stock?"].map(parse_stock).to_numpy(),
                              index=_text(var, "Parent").str.removeprefix("id:")).groupby(level=0).agg(_any_stock)
        blank = _text(out, "In stock?").eq("")
        parent_stock = _text(out, "ID").map(var_stock).astype(object).combine_first(   # parent named by id or SKU
            _text(out, "SKU").map(var_stock).astype(object))
        fill = blank & parent_stock.notna()
        out.loc[fill, "In stock?"] = parent_stock[fill].map(lambda v: "1" if v else "0")
    no_price = _text(out, "Sale price").eq("") & _text(out, "Regular price").eq("")
    parent_price = _text(out, "ID").map(cheapest).fillna(_text(out, "SKU").map(cheapest))
    fill = no_price & parent_price.notna()
    out.loc[fill, "Regular price"] = parent_price[fill].map(lambda v: f"{v:.2f}")
    return out


def _clean_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():      # JSON ids with gaps arrive as float64
        value = int(value)
    return str(value).strip() or None


def import_feed(feed: pd.DataFrame, preset: str = "generic",
                mapping: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Map a feed to the canonical schema and clean it (spec 5.2 rules that apply to a store's own listings).

    Returns the catalog and a log of rows removed per rule.
    """
    columns = {f: list(PRESETS[preset].get(f, [])) for f in FIELDS}
    columns.update({f: list(c) for f, c in OPTIONAL_FIELDS.items()})
    for field, column in (mapping or {}).items():
        if field not in columns:
            raise ValueError(f"unknown field {field!r}; expected one of {FIELDS + list(OPTIONAL_FIELDS)}")
        columns[field] = [column]
    missing = [f for f in ("product_id", "title") if not any(c in feed.columns for c in columns[f])]
    if missing:
        raise ValueError(f"feed has no column for {missing}; columns are {list(feed.columns)}; pass a mapping")

    def pick(field: str) -> pd.Series:
        out = pd.Series([None] * len(feed), index=feed.index, dtype=object)
        for column in columns[field]:
            if column in feed.columns:
                values = feed[column].where(feed[column].astype(str).str.strip().ne(""))
                out = out.fillna(values)
        return out

    log: dict[str, Any] = {"raw_rows": len(feed)}
    feed = drop_unpublished(feed, log)
    feed = merge_woo_variations(feed, log) if preset == "woocommerce" else feed
    log.setdefault("dropped_variations", 0)
    df = pd.DataFrame({f: pick(f) for f in FIELDS})
    rating = pd.to_numeric(pick("rating"), errors="coerce")
    df["avg_rating"] = rating.where(rating.between(1, 5))                  # stars 1-5; anything else unknown
    counts = pd.to_numeric(pick("rating_count"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["rating_count"] = counts.fillna(0).clip(lower=0).astype(int)      # odd values (text, inf) count as 0
    df["product_id"] = df["product_id"].map(_clean_id)
    df["in_stock"] = row_stock(feed, preset, (mapping or {}).get("in_stock"))
    stock_by_id = df.groupby("product_id")["in_stock"].agg(_any_stock)        # before variant rows are dropped
    for field in ("title", "description", "features", "brand"):
        df[field] = df[field].map(lambda v: clean_text(v) if isinstance(v, str) else "")   # NaN/None -> ""
    multi = preset == "woocommerce"
    df["category"] = df["category"].map(lambda v: last_level(v, multi) if isinstance(v, str) and v.strip() else "")
    df["price"] = df["price"].map(parse_price).astype(float)

    before = len(df)
    df = df[df["product_id"].notna() & df["title"].str.len().gt(0)]     # e.g. Shopify image-only variant rows
    log["dropped_no_id_or_title"] = before - len(df)
    before = len(df)
    df = df.drop_duplicates("product_id", keep="first")
    log["dropped_duplicate_id"] = before - len(df)

    df["in_stock"] = [_any_stock(pd.Series([v], dtype=object)) for v in df["product_id"].map(stock_by_id)]
    df["brand"] = df["brand"].replace("", None)
    df["category"] = df["category"].replace("", "Other")
    df = df.assign(popularity=0).reset_index(drop=True)
    df["product_text"] = df.apply(build_product_text, axis=1)
    log.update(out_of_stock=sum(v is False for v in df["in_stock"]),
               stock_unknown=sum(v is None for v in df["in_stock"]))
    log.update(products=len(df), price_missing=int(df["price"].isna().sum()),
               brand_missing=int(df["brand"].isna().sum()), n_categories=int(df["category"].nunique()))
    return df, log


def build_store(name: str, source: str | Path, preset: str = "generic", mapping: dict[str, str] | None = None,
                model: Any = None, cfg: dict | None = None) -> dict:
    """Import a feed and save the catalog and its embeddings under stores/<name>/ (loadable by load_resources)."""
    check_store_name(name)
    if not str(source).lower().startswith(("http://", "https://")):
        source = Path(source).resolve()               # absolute, so a later --refresh works from any folder
    return build_store_from_feed(name, read_feed(source), preset, mapping, model, cfg, source=str(source))


def build_store_from_feed(name: str, feed: pd.DataFrame, preset: str = "generic",
                          mapping: dict[str, str] | None = None, model: Any = None, cfg: dict | None = None,
                          source: str = "upload", site: str | None = None) -> dict:
    """Same as build_store for a feed already read (the demo app's upload and store-link flows)."""
    from src.embed import encode_products, load_model

    check_store_name(name)
    cfg = cfg or load_config()
    catalog, log = import_feed(feed, preset, mapping)
    if catalog.empty:
        raise ValueError("no products left after import")
    out = STORES_DIR / name
    old_vectors, old_texts = previous_vectors(out, cfg["embedding"]["model"])
    keys = list(zip(catalog["product_id"], catalog["product_text"]))
    todo = [i for i, k in enumerate(keys) if k not in old_vectors]           # new products or changed text
    fresh = {}
    if todo:
        model = model or load_model(cfg["embedding"]["model"])
        enc = encode_products([keys[i][1] for i in todo], model, batch_size=cfg["embedding"]["batch_size"])
        fresh = {keys[i]: v for i, v in zip(todo, enc)}
    emb = np.vstack([fresh.get(k, old_vectors.get(k)) for k in keys]).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    save_store_files(out, catalog, emb)
    ids = set(catalog["product_id"])
    log.update(added=len(ids - set(old_texts)), removed=len(set(old_texts) - ids),
               changed=sum(pid in old_texts and old_texts[pid] != text for pid, text in keys),
               reused=len(keys) - len(todo), updated_at=time.strftime("%Y-%m-%d %H:%M"))
    log.update(store=name, source=source, preset=preset, mapping=mapping or {}, site=site_host(site),
               embedding_model=cfg["embedding"]["model"])
    (out / "import_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def save_store_files(out: Path, catalog: pd.DataFrame, emb: np.ndarray) -> None:
    """Write embeddings, ids and catalog to temporary files, then swap them in (catalog last: its change makes the
    API reload). If a swap fails, the previous files are put back, so the store never mixes two imports."""
    files = {"embeddings.npy": emb.astype(np.float32),
             "product_ids.npy": np.asarray(catalog["product_id"].tolist(), dtype=str)}
    for name, arr in files.items():
        with open(out / f"{name}.tmp", "wb") as f:
            np.save(f, arr)
    catalog.to_pickle(out / "catalog.pkl.tmp")
    moved = []
    try:
        for name in (*files, "catalog.pkl"):
            if name != "catalog.pkl" and (out / name).exists():
                os.replace(out / name, out / f"{name}.bak")
                moved.append(name)
            os.replace(out / f"{name}.tmp", out / name)
    except Exception:
        for name in moved:
            os.replace(out / f"{name}.bak", out / name)
        raise
    finally:
        for leftover in [*out.glob("*.tmp"), *out.glob("*.bak")]:
            leftover.unlink(missing_ok=True)


def previous_vectors(store_dir: Path, model_name: str) -> tuple[dict, dict]:
    """A store's saved vectors keyed by (product_id, product_text), and its texts by id, for an incremental update.
    Empty when there is no earlier import or it used another embedding model."""
    from src.embed import load_embeddings

    try:
        log = json.loads((store_dir / "import_log.json").read_text(encoding="utf-8"))
        catalog = pd.read_pickle(store_dir / "catalog.pkl")
        emb, ids = load_embeddings(store_dir)
    except Exception:  # noqa: BLE001 - missing or corrupt files: embed everything again
        return {}, {}
    if log.get("embedding_model") != model_name or len(emb) != len(catalog) \
            or list(ids) != [str(p) for p in catalog["product_id"]]:
        return {}, {}
    texts = dict(zip(catalog["product_id"], catalog["product_text"]))
    return {(pid, text): emb[i] for i, (pid, text) in enumerate(texts.items())}, texts


def refresh_store(name: str, model: Any = None, cfg: dict | None = None) -> dict:
    """Re-read a store from the source it was imported from (file path, feed URL or Shopify address) and update it;
    only products whose text changed are embedded again. Run it on a schedule to keep stock and prices current."""
    check_store_name(name)
    log_path = STORES_DIR / name / "import_log.json"
    if not log_path.exists():
        raise ValueError(f"no store {name!r}; import it first")
    prev = json.loads(log_path.read_text(encoding="utf-8"))
    source = str(prev.get("source") or "")
    if source in ("", "upload"):
        raise ValueError("this store was added from an uploaded file; upload the new export to update it")
    url = source
    if prev.get("preset") == "shopify_json" and not source.lower().split("?")[0].endswith(".json"):
        url = shopify_feed_url(source)                # the Shopify address typed in the app, with or without https
    elif not source.lower().startswith(("http://", "https://")) and not Path(source).exists():
        raise ValueError(f"the original file {source} is no longer there; upload the new export to update it")
    return build_store_from_feed(name, read_feed(url), prev.get("preset", "generic"), prev.get("mapping") or None,
                                 model, cfg, source=source, site=prev.get("site"))


def detect_preset(feed: pd.DataFrame) -> str | None:
    """Guess the export format from its columns; None when no preset fits (ask the user to map columns)."""
    cols = set(feed.columns)
    if {"Handle", "Title"} <= cols:
        return "shopify"
    if {"handle", "title"} <= cols or {"id", "title", "vendor"} <= cols:
        return "shopify_json"
    if "Name" in cols and ({"Regular price", "Sale price"} & cols or "SKU" in cols):
        return "woocommerce"
    if {"product_id", "title"} <= cols:
        return "generic"
    return None


def store_name_from(text: str) -> str:
    """Turn any label ('My Shop!', 'https://shop.example.com') into a valid store name."""
    text = re.sub(r"^https?://", "", str(text).strip().lower()).split("/")[0]
    name = re.sub(r"[^a-z0-9_-]+", "-", text).strip("-")[:64]
    return name or "mystore"


def shopify_feed_url(address: str) -> str:
    """'mystore.com' or 'https://mystore.com/collections/x?y' -> 'https://mystore.com/products.json'.

    Only public domain names are accepted (no IP addresses, localhost or intranet names), so the link can only
    reach a real online store.
    """
    import ipaddress

    host = re.split(r"[/?#]", re.sub(r"^https?://", "", address.strip(), flags=re.I))[0].lower()
    host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host          # drop a port
    try:
        ipaddress.ip_address(host.strip("[]"))
        is_ip = True
    except ValueError:
        is_ip = False
    if not host or is_ip or "." not in host or host.endswith((".local", ".localhost", ".internal"))             or not re.fullmatch(r"[a-z0-9.-]+", host):
        raise ValueError("enter the store's public web address, e.g. mystore.com")
    return f"https://{host}/products.json"


def site_host(address: str | None) -> str | None:
    """'https://MyStore.com/shop' -> 'mystore.com' (None when empty or not a public domain name)."""
    if not address or not address.strip():
        return None
    try:
        return shopify_feed_url(address).removeprefix("https://").removesuffix("/products.json")
    except ValueError:
        return None


def link_template(store: str | None) -> str | None:
    """Product-page link pattern for the chat's 'buy' hand-off, e.g. 'https://shop.com/products/{id}'.

    Benchmark catalog: Amazon product pages (ids are ASINs). Imported Shopify stores: /products/<handle> when the
    store's address is known. Otherwise None (the chat then names the product instead of linking).
    """
    if store is None:
        return "https://www.amazon.com/dp/{id}"
    if not valid_store_name(store):
        return None
    log_path = STORES_DIR / store / "import_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    site = log.get("site") or site_host(str(log.get("source", "")))      # e.g. the Shopify address typed in
    if site and log.get("preset") in ("shopify", "shopify_json"):
        return f"https://{site}/products/{{id}}"
    return None


def list_stores() -> list[str]:
    return sorted(p.name for p in STORES_DIR.glob("*") if (p / "catalog.pkl").exists())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Import a store's product export (CSV/JSON, path or URL).")
    parser.add_argument("--name", required=True, help="store name, used as stores/<name>/")
    parser.add_argument("--file", help="CSV or JSON export, path or URL (e.g. https://shop/products.json)")
    parser.add_argument("--refresh", action="store_true",
                        help="update an imported store from its saved source (only changed products are re-embedded)")
    parser.add_argument("--every", type=float, default=0, help="with --refresh: repeat every N minutes")
    parser.add_argument("--preset", default="generic", choices=sorted(PRESETS))
    parser.add_argument("--map", default="", help="field=column pairs, e.g. 'title=Name,price=Cost'")
    args = parser.parse_args(argv)
    if args.every < 0:
        parser.error("--every must be 0 or more minutes")
    if args.refresh and (args.file or args.map):
        parser.error("--refresh re-reads the saved source with the saved settings; --file and --map are not used")
    if args.refresh:
        while True:
            try:
                log = refresh_store(args.name)
                print(f"{log['updated_at']} {args.name}: {log['products']} products ({log['added']} added, "
                      f"{log['removed']} removed, {log['changed']} changed, {log['out_of_stock']} out of stock)")
            except Exception as exc:  # noqa: BLE001 - a scheduled update keeps running after one failed read
                print(f"update failed: {exc}")
                if not args.every:
                    raise SystemExit(1) from None
            if not args.every:
                return
            time.sleep(args.every * 60)
    if not args.file:
        parser.error("--file is required (or use --refresh for an imported store)")
    mapping = dict(pair.split("=", 1) for pair in args.map.split(",") if pair)
    print(json.dumps(build_store(args.name, args.file, args.preset, mapping), indent=2))


if __name__ == "__main__":
    main()
