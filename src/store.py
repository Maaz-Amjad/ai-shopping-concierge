"""Store connector: import any store's product export into the canonical schema and build its search index.

Presets map common export formats to the spec 5.1 fields; `mapping` overrides or adds columns for anything else.
Read-only: the store's own data is never modified. Output: stores/<name>/ (catalog.pkl, embeddings, import_log.json).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src import ROOT, load_config
from src.data import build_product_text, clean_text, parse_price

STORES_DIR = ROOT / "stores"
STORE_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
PAGE_SIZE, MAX_PAGES = 250, 200     # Shopify products.json pagination
FIELDS = ["product_id", "title", "brand", "category", "price", "description", "features"]

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
    return {**{k: product.get(k) for k in ("id", "handle", "title", "vendor", "product_type", "body_html")},
            "price": variants[0].get("price"), "tags": "; ".join(tags) if isinstance(tags, list) else tags}


def last_level(category: str, multi: bool = False) -> str:
    """Deepest level of a category path. multi=True (WooCommerce lists several categories with commas):
    'Kitchen > Kettles, Gifts' -> 'Kettles'. Otherwise commas belong to level names ('Coffee, Tea & Cocoa')."""
    path = str(category).split(",")[0] if multi else str(category)
    return clean_text(path.split(">")[-1])


def _text(feed: pd.DataFrame, column: str) -> pd.Series:
    return feed[column].astype(str).str.strip() if column in feed else pd.Series("", index=feed.index)


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
    for field, column in (mapping or {}).items():
        if field not in FIELDS:
            raise ValueError(f"unknown field {field!r}; expected one of {FIELDS}")
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
    df["product_id"] = df["product_id"].map(_clean_id)
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

    df["brand"] = df["brand"].replace("", None)
    df["category"] = df["category"].replace("", "Other")
    df = df.assign(popularity=0).reset_index(drop=True)
    df["product_text"] = df.apply(build_product_text, axis=1)
    log.update(products=len(df), price_missing=int(df["price"].isna().sum()),
               brand_missing=int(df["brand"].isna().sum()), n_categories=int(df["category"].nunique()))
    return df, log


def build_store(name: str, source: str | Path, preset: str = "generic", mapping: dict[str, str] | None = None,
                model: Any = None, cfg: dict | None = None) -> dict:
    """Import a feed and save the catalog and its embeddings under stores/<name>/ (loadable by load_resources)."""
    check_store_name(name)
    return build_store_from_feed(name, read_feed(source), preset, mapping, model, cfg, source=str(source))


def build_store_from_feed(name: str, feed: pd.DataFrame, preset: str = "generic",
                          mapping: dict[str, str] | None = None, model: Any = None, cfg: dict | None = None,
                          source: str = "upload") -> dict:
    """Same as build_store for a feed already read (the demo app's upload and store-link flows)."""
    from src.embed import encode_products, load_model, save_embeddings

    check_store_name(name)
    cfg = cfg or load_config()
    catalog, log = import_feed(feed, preset, mapping)
    if catalog.empty:
        raise ValueError("no products left after import")
    model = model or load_model(cfg["embedding"]["model"])
    emb = encode_products(catalog["product_text"].tolist(), model, batch_size=cfg["embedding"]["batch_size"])
    out = STORES_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    catalog.to_pickle(out / "catalog.pkl")
    save_embeddings(emb, catalog["product_id"].tolist(), out)
    log.update(store=name, source=source, preset=preset, mapping=mapping or {})
    (out / "import_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


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
    """'mystore.com' or 'https://mystore.com/collections/x' -> 'https://mystore.com/products.json'."""
    host = re.sub(r"^https?://", "", address.strip()).split("/")[0]
    if not host:
        raise ValueError("enter the store's web address, e.g. mystore.com")
    return f"https://{host}/products.json"


def list_stores() -> list[str]:
    return sorted(p.name for p in STORES_DIR.glob("*") if (p / "catalog.pkl").exists())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Import a store's product export (CSV/JSON, path or URL).")
    parser.add_argument("--name", required=True, help="store name, used as stores/<name>/")
    parser.add_argument("--file", required=True, help="CSV or JSON export, path or URL (e.g. https://shop/products.json)")
    parser.add_argument("--preset", default="generic", choices=sorted(PRESETS))
    parser.add_argument("--map", default="", help="field=column pairs, e.g. 'title=Name,price=Cost'")
    args = parser.parse_args(argv)
    mapping = dict(pair.split("=", 1) for pair in args.map.split(",") if pair)
    print(json.dumps(build_store(args.name, args.file, args.preset, mapping), indent=2))


if __name__ == "__main__":
    main()
