"""Store connector: import any store's product export into the canonical schema and build its search index.

Presets map common export formats to the spec 5.1 fields; `mapping` overrides or adds columns for anything else.
Read-only: the store's own data is never modified. Output: stores/<name>/ (catalog.pkl, embeddings, import_log.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src import ROOT, load_config
from src.data import build_product_text, clean_text, parse_price

STORES_DIR = ROOT / "stores"
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


def read_feed(source: str | Path) -> pd.DataFrame:
    """Read a CSV or JSON export from a path or URL. Shopify products.json is flattened to one row per product."""
    source = str(source)
    if source.lower().split("?")[0].endswith(".json"):
        if source.startswith(("http://", "https://")):
            import httpx

            data: Any = httpx.get(source, timeout=30, follow_redirects=True).raise_for_status().json()
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


def last_level(category: str) -> str:
    """'Home > Kitchen > Kettles, Gifts' -> 'Kettles' (first category, deepest level)."""
    return clean_text(str(category).split(",")[0].split(">")[-1])


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

    df = pd.DataFrame({f: pick(f) for f in FIELDS})
    log: dict[str, Any] = {"raw_rows": len(df)}
    df["product_id"] = df["product_id"].map(lambda v: None if v is None or pd.isna(v) else str(v).strip())
    for field in ("title", "description", "features", "brand"):
        df[field] = df[field].map(lambda v: clean_text(v) if isinstance(v, str) else "")   # NaN/None -> ""
    df["category"] = df["category"].map(lambda v: last_level(v) if isinstance(v, str) and v.strip() else "")
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
    from src.embed import encode_products, load_model, save_embeddings

    cfg = cfg or load_config()
    catalog, log = import_feed(read_feed(source), preset, mapping)
    if catalog.empty:
        raise ValueError("no products left after import")
    model = model or load_model(cfg["embedding"]["model"])
    emb = encode_products(catalog["product_text"].tolist(), model, batch_size=cfg["embedding"]["batch_size"])
    out = STORES_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    catalog.to_pickle(out / "catalog.pkl")
    save_embeddings(emb, catalog["product_id"].tolist(), out)
    log.update(store=name, source=str(source), preset=preset, mapping=mapping or {})
    (out / "import_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


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
