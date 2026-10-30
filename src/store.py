"""Store connector: import any store's product export into the canonical schema and build its search index.

Presets map common export formats to the spec 5.1 fields; `mapping` overrides or adds columns for anything else.
Read-only: the store's own data is never modified. Output: stores/<name>/ (catalog.pkl, embeddings, import_log.json).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
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
    if isinstance(tags, str):                      # webhooks send "a, b"; products.json a list: same text either way
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    known = [bool(v["available"]) for v in variants if v.get("available") is not None]
    return {**{k: product.get(k) for k in ("id", "handle", "title", "vendor", "product_type", "body_html",
                                                        "updated_at")},
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
    cheapest = pd.Series(price.map(parse_price).to_numpy(), index=_text(var, "Parent").str.removeprefix("id:"))
    cheapest = cheapest.groupby(level=0).min()
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
    columns["source_id"] = ["id", "ID"]                 # the platform's own id (webhook deletes carry only this)
    columns["source_updated_at"] = ["updated_at", "date_modified_gmt"]   # when the platform last changed it
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
    df["source_id"] = pick("source_id").map(_clean_id)
    df["source_updated_at"] = pd.to_datetime(pick("source_updated_at"), errors="coerce", utc=True)
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
    df["product_text"] = [build_product_text(r) for _, r in df.iterrows()]   # also fine when nothing is left
    log.update(catalog_stats(df))
    return df, log


def catalog_stats(df: pd.DataFrame) -> dict:
    """Counts shown after an import or update."""
    stock = df["in_stock"] if "in_stock" in df else pd.Series([None] * len(df), dtype=object)
    return {"out_of_stock": sum(v is False or v is np.False_ for v in stock),
            "stock_unknown": sum(v is None for v in stock), "products": len(df),
            "price_missing": int(df["price"].isna().sum()), "brand_missing": int(df["brand"].isna().sum()),
            "n_categories": int(df["category"].nunique())}


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

    check_store_name(name)
    catalog, log = import_feed(feed, preset, mapping)
    if catalog.empty:
        raise ValueError("no products left after import")
    log.update(source=source, preset=preset, mapping=mapping or {}, site=site_host(site))
    with store_lock(name):
        previous = STORES_DIR / name / "catalog.pkl"
        if previous.exists():
            try:
                catalog = keep_newer_rows(catalog, pd.read_pickle(previous))
            except Exception:  # noqa: BLE001 - an unreadable old catalog is simply replaced
                pass
        return write_store(name, catalog, log, model, cfg)


def write_store(name: str, catalog: pd.DataFrame, log: dict, model: Any = None, cfg: dict | None = None) -> dict:
    """Save a store's catalog and embeddings, embedding only new products or changed texts; returns the log with
    added/removed/changed/reused counts (call with the store's lock held). Popularity comes from shopper events."""
    from src.embed import encode_products, load_model

    cfg = cfg or load_config()
    out = STORES_DIR / name
    catalog = catalog.copy()
    catalog["popularity"] = catalog["product_id"].map(event_popularity(name)).fillna(0).astype(int)
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
    log.update(store=name, embedding_model=cfg["embedding"]["model"])
    (out / "import_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


_store_locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)
STALE_LOCK_SECONDS = 600


@contextmanager
def store_lock(name: str, timeout: float = 60.0):
    """One writer per store across threads and processes (API webhooks, scheduled --refresh, the app): a lock file
    created exclusively in the store folder; a lock older than 10 minutes is treated as left over from a crash."""
    path = STORES_DIR / name / ".lock"
    deadline = time.monotonic() + timeout
    if not _store_locks[name].acquire(timeout=timeout):
        raise TimeoutError(f"store {name!r} is being updated; try again shortly")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    if time.time() - path.stat().st_mtime > STALE_LOCK_SECONDS:
                        path.unlink(missing_ok=True)
                        continue
                except OSError:
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"store {name!r} is being updated; try again shortly") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            path.unlink(missing_ok=True)
    finally:
        _store_locks[name].release()


def shopify_webhook_feed(product: dict) -> pd.DataFrame:
    """A Shopify products/create|update webhook payload -> a one-row feed for the shopify_json preset. Webhook
    variants carry inventory fields instead of `available`; a product that is not active is dropped as unpublished."""
    row = flatten_shopify(product)
    known = [v for v in product.get("variants") or []
             if "inventory_management" in v or "inventory_quantity" in v]    # fields missing = stock unknown
    if row["available"] is None and known:
        row["available"] = any(not v.get("inventory_management") or v.get("inventory_policy") == "continue"
                               or (v.get("inventory_quantity") or 0) > 0 for v in known)
    hidden = "published_at" in product and product["published_at"] is None   # not on the online store
    row["Status"] = "unpublished" if hidden else (product.get("status") or "")
    return pd.DataFrame([row])


def woocommerce_webhook_feed(product: dict) -> pd.DataFrame:
    """A WooCommerce product.created|updated|restored webhook payload -> a one-row feed in the export's columns."""
    names = lambda items: ", ".join(str(i.get("name", "")) for i in items or [] if isinstance(i, dict))  # noqa: E731
    stock = {"instock": "1", "onbackorder": "1", "outofstock": "0"}.get(str(product.get("stock_status")), "")
    return pd.DataFrame([{
        "ID": product.get("id"), "SKU": product.get("sku") or "", "Type": product.get("type") or "",
        "Name": product.get("name") or "", "Description": product.get("description") or "",
        "Short description": product.get("short_description") or "",
        "Regular price": str(product.get("regular_price") or product.get("price") or ""),
        "Sale price": str(product.get("sale_price") or ""), "Categories": names(product.get("categories")),
        "Tags": names(product.get("tags")), "In stock?": stock, "Brands": names(product.get("brands")),
        "average_rating": product.get("average_rating"), "rating_count": product.get("rating_count"),
        "date_modified_gmt": product.get("date_modified_gmt"),
        "Published": "1" if product.get("status") == "publish" else "0"}])


TOMBSTONE_DAYS = 30


def _stored_times(catalog: pd.DataFrame) -> tuple[dict, dict]:
    """Stored platform times by product id and by platform id (rows without a platform id are not matched by it)."""
    if "source_updated_at" not in catalog:
        return {}, {}
    stored = pd.to_datetime(catalog["source_updated_at"], errors="coerce", utc=True)
    by_id = dict(zip(catalog["product_id"], stored))
    sids = catalog["source_id"] if "source_id" in catalog else pd.Series([None] * len(catalog), index=catalog.index)
    by_sid = {str(s): t for s, t in zip(sids, stored) if s is not None and pd.notna(s)}
    return by_id, by_sid


def _lookup(table_id: dict, table_sid: dict, pid: Any, sid: Any) -> Any:
    found = table_id.get(pid)
    if (found is None or pd.isna(found)) and sid is not None and pd.notna(sid):
        found = table_sid.get(str(sid))
    return None if found is None or pd.isna(found) else found


def drop_stale(new: pd.DataFrame, catalog: pd.DataFrame,
               tombstones: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Skip webhook rows older than the stored version (the platform's updated_at is inside the signed body), and rows
    for a deleted product unless they are newer than its deletion: a replayed or late delivery cannot restore an old
    price, stock state or a deleted product. A row without a time keeps the stored time."""
    if new.empty or "source_updated_at" not in new:
        return new, []
    by_id, by_sid = _stored_times(catalog)
    dead = {k: pd.Timestamp(v) for k, v in (tombstones or {}).items()}
    new, stale = new.copy(), []
    for i, row in new.iterrows():
        before = _lookup(by_id, by_sid, row["product_id"], row.get("source_id"))
        deleted = dead.get(str(row["product_id"])) or (dead.get(str(row["source_id"]))
                                                       if pd.notna(row.get("source_id")) else None)
        when = row["source_updated_at"]
        if pd.isna(when):
            if deleted is not None:
                stale.append(row["product_id"])
            elif before is not None:
                new.at[i, "source_updated_at"] = before
        elif (before is not None and when < before) or (deleted is not None and when <= deleted):
            stale.append(row["product_id"])
    return new[~new["product_id"].isin(stale)], stale


def keep_newer_rows(fresh: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """A full refresh (possibly a cached snapshot) never replaces a product a newer webhook already updated."""
    if "source_updated_at" not in fresh or "source_updated_at" not in previous or previous.empty:
        return fresh
    old = previous.drop_duplicates("product_id").set_index("product_id")
    old_t = pd.to_datetime(old["source_updated_at"], errors="coerce", utc=True)
    rows = []
    for _, row in fresh.iterrows():
        pid = row["product_id"]
        if pid in old.index and pd.notna(old_t.get(pid)) and pd.notna(row["source_updated_at"]) \
                and old_t[pid] > row["source_updated_at"]:
            rows.append(old.loc[pid].to_dict() | {"product_id": pid})
        else:
            rows.append(row.to_dict())
    return pd.DataFrame(rows, columns=fresh.columns)


def _tombstones(out: Path) -> dict:
    path = out / "tombstones.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        data = {}
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=TOMBSTONE_DAYS)
    return {k: v for k, v in data.items() if pd.Timestamp(v) > cutoff}


def keep_known_fields(new: pd.DataFrame, catalog: pd.DataFrame, same_source: bool) -> pd.DataFrame:
    """A webhook payload may lack fields the original import had (brand, ratings) or take them from another field
    (a Shopify CSV's "Product Category" vs the webhook's product_type): keep the stored values so hard filters keep
    matching catalog values. The full refresh still replaces them."""
    if new.empty or catalog.empty:
        return new
    new = new.copy()
    by_id = catalog.drop_duplicates("product_id").set_index("product_id")
    sid = catalog.dropna(subset=["source_id"]).drop_duplicates("source_id").set_index("source_id") \
        if "source_id" in catalog else None
    for i, row in new.iterrows():
        if row["product_id"] in by_id.index:
            old = by_id.loc[row["product_id"]]
        elif sid is not None and pd.notna(row.get("source_id")) and row["source_id"] in sid.index:
            old = sid.loc[row["source_id"]]
        else:
            continue
        if not row.get("brand") or pd.isna(row.get("brand")):
            new.at[i, "brand"] = old.get("brand")
        if pd.isna(row.get("avg_rating")) and pd.notna(old.get("avg_rating")):
            new.at[i, "avg_rating"], new.at[i, "rating_count"] = old.get("avg_rating"), old.get("rating_count", 0)
        if not same_source:
            new.at[i, "category"] = old.get("category")
    new["product_text"] = [build_product_text(r) for _, r in new.iterrows()]
    return new


def apply_store_changes(name: str, feed: pd.DataFrame | None = None, delete_ids: list[str] | tuple = (),
                        model: Any = None, cfg: dict | None = None, preset: str | None = None) -> dict:
    """Webhook update: add or replace the products in `feed` (in `preset` format; webhooks arrive as shopify_json
    or woocommerce whatever file the store was first imported from; default: the store's own preset and mapping),
    remove `delete_ids` (product id or the platform's id) and any feed product that is no longer published.
    Only products whose text changed are embedded again."""
    check_store_name(name)
    out = STORES_DIR / name
    if not (out / "catalog.pkl").exists():
        raise ValueError(f"no store {name!r}; import it first")
    with store_lock(name):
        log = json.loads((out / "import_log.json").read_text(encoding="utf-8"))
        catalog = pd.read_pickle(out / "catalog.pkl")
        if "source_id" not in catalog:                     # imported before Phase 13.4
            catalog["source_id"] = None
        own = log.get("preset", "generic")
        gone = {str(i) for i in delete_ids}
        new = catalog.iloc[:0]
        log["stale"] = []
        tombstones = _tombstones(out)
        by_id, by_sid = _stored_times(catalog)
        now = pd.Timestamp.now(tz="UTC")
        if feed is not None and not feed.empty:
            fmt = preset or own                            # a saved mapping belongs to the store's own files only
            new, _ = import_feed(feed, fmt, (log.get("mapping") or None) if preset is None else None)
            when = pd.to_datetime(feed[[c for c in ("updated_at", "date_modified_gmt") if c in feed][0]]
                                  if {"updated_at", "date_modified_gmt"} & set(feed) else pd.Series(
                                      [None] * len(feed), index=feed.index), errors="coerce", utc=True)
            hidden = set()                                 # unpublished in this (signed, timed) message
            for c in ("id", "ID", "handle", "Handle", "SKU", "product_id"):
                if c not in feed:
                    continue
                for v, t in zip(feed[c], when):
                    key = _clean_id(v)
                    if key is None or str(key) in set(new["product_id"]) | set(new["source_id"].dropna()):
                        continue
                    stored = _lookup(by_id, by_sid, str(key), str(key))
                    if stored is not None and pd.notna(t) and t < stored:
                        continue                           # an older "unpublish" never removes a newer product
                    hidden.add(str(key))
                    tombstones[str(key)] = str(max(x for x in (t, stored, now) if x is not None and pd.notna(x)))
            gone |= hidden
            new, log["stale"] = drop_stale(new, catalog, tombstones)
            new = keep_known_fields(new, catalog, same_source=fmt == own)
            for _, row in new.iterrows():                  # back after a deletion (newer than it): forget it
                tombstones.pop(str(row["product_id"]), None)
                if pd.notna(row.get("source_id")):
                    tombstones.pop(str(row["source_id"]), None)
        pid, sid = catalog["product_id"].astype(str), catalog["source_id"].astype(str)
        new_ids, new_sids = set(new["product_id"]), set(new["source_id"].dropna().astype(str))
        drop = pid.isin(gone | new_ids) | sid.isin(gone | new_sids)      # also a product renamed (same platform id)
        unmatched = [str(i) for i in delete_ids if not (pid == str(i)).any() and not (sid == str(i)).any()]
        log["not_found"], log["applied"] = [], len(new)
        if unmatched and own in ("shopify", "shopify_json") and log.get("site"):
            try:                                          # a delete carries only Shopify's id: ask the live store
                live = fetch_products_json(shopify_feed_url(log["site"]))
                handles = {str(p.get("handle")): str(p.get("id")) for p in (live or {}).get("products", [])
                           if isinstance(p, dict) and p.get("handle")}
            except Exception:  # noqa: BLE001 - the next refresh catches up
                handles = {}
            if handles:                                   # never empty the store on a failed or empty read
                drop |= ~pid.isin(set(handles) | new_ids)
                catalog["source_id"] = pid.map(handles).where(pid.isin(set(handles)), catalog["source_id"])
            else:
                log["not_found"] = unmatched
        elif unmatched:
            log["not_found"] = unmatched                  # e.g. a CSV store without its address: re-upload later
        for _, row in catalog[drop & ~pid.isin(new_ids)].iterrows():     # remember deletions (both ids)
            stored = _lookup(by_id, by_sid, row["product_id"], row.get("source_id"))
            mark = str(max(stored, now) if stored is not None else now)
            tombstones[str(row["product_id"])] = mark
            if pd.notna(row.get("source_id")):
                tombstones[str(row["source_id"])] = mark
        (out / "tombstones.json").write_text(json.dumps(tombstones), encoding="utf-8")
        if new.empty and not drop.any():                  # nothing to change (e.g. a stale delivery): no rewrite
            return log
        merged = pd.concat([catalog[~drop], new], ignore_index=True)
        if merged.empty:
            raise ValueError("the update would leave the store with no products")
        log.update(catalog_stats(merged))
        return write_store(name, merged, log, model, cfg)


# ---- Phase 13.8: shopper events (feedback loop) -----------------------------------------------------------

EVENT_WEIGHTS = {"buy": 5, "save": 3, "click": 1, "shown": 0}   # popularity points per browser, product and day
MAX_EVENTS = 50                                                  # kept per request
EVENT_DAYS = 365                                                 # events older than this are ignored
_known: dict[str, tuple[float, set]] = {}


def _known_ids(name: str) -> set:
    path = STORES_DIR / name / "catalog.pkl"
    stamp = path.stat().st_mtime
    if name not in _known or _known[name][0] != stamp:
        _known[name] = (stamp, set(pd.read_pickle(path)["product_id"].astype(str)))
    return _known[name][1]


def record_events(name: str, client_id: str, events: list) -> int:
    """Append shopper events (shown / click / save / buy of a catalog product) to stores/<name>/events.jsonl.
    Only the type, product id, a random per-browser id and the server's time are kept: no query text, no IP."""
    check_store_name(name)
    if not (STORES_DIR / name / "catalog.pkl").exists():
        raise ValueError(f"no store {name!r}; import it first")
    known = _known_ids(name)
    client = re.sub(r"[^A-Za-z0-9_-]", "", str(client_id))[:64] or "anonymous"
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    rows = [{"ts": ts, "client": client, "type": e["type"], "product_id": e["product_id"]}
            for e in list(events)[:4 * MAX_EVENTS] if isinstance(e, dict) and e.get("type") in EVENT_WEIGHTS
            and isinstance(e.get("product_id"), str) and e["product_id"] in known][:MAX_EVENTS]
    if rows:
        with _store_locks[f"{name}:events"]:
            with open(STORES_DIR / name / "events.jsonl", "a", encoding="utf-8") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def _read_events(name: str, days: int) -> pd.DataFrame:
    path = STORES_DIR / name / "events.jsonl"
    cols = ["ts", "client", "type", "product_id"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue                                  # a torn last line from a crash is skipped
    df = pd.DataFrame(rows, columns=cols)
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - days * 86400))
    return df[df["ts"] >= cutoff]


def _scores(df: pd.DataFrame) -> pd.Series:
    """Weighted points per product, each (browser, action, product) counted once a day: repeats cannot inflate it."""
    if df.empty:
        return pd.Series(dtype=int)
    once = df.assign(day=df["ts"].str[:10]).drop_duplicates(["client", "type", "product_id", "day"])
    points = once["type"].map(EVENT_WEIGHTS).fillna(0).astype(int).groupby(once["product_id"]).sum()
    return points[points > 0].sort_values(ascending=False)


def event_popularity(name: str, days: int = EVENT_DAYS) -> pd.Series:
    """Popularity for an imported store: buy 5, save 3, click 1 points (shown 0), per browser and day."""
    return _scores(_read_events(name, days))


def update_popularity(name: str, model: Any = None, cfg: dict | None = None) -> dict:
    """Recompute popularity from events and save the store (no product is embedded again)."""
    check_store_name(name)
    out = STORES_DIR / name
    if not (out / "catalog.pkl").exists():
        raise ValueError(f"no store {name!r}; import it first")
    with store_lock(name):
        log = json.loads((out / "import_log.json").read_text(encoding="utf-8"))
        return write_store(name, pd.read_pickle(out / "catalog.pkl"), log, model, cfg)


def event_summary(name: str, days: int = 30) -> dict:
    """What shoppers did in the last `days` days, for the store owner (the coming dashboard)."""
    df = _read_events(name, days)
    counts = {t: int((df["type"] == t).sum()) for t in ("shown", "click", "save", "buy")}
    rate = (lambda n: round(n / counts["shown"], 4) if counts["shown"] else None)
    top = _scores(df).head(10)
    return {"days": days, "counts": counts, "shoppers": int(df["client"].nunique()),
            "click_rate": rate(counts["click"]), "save_rate": rate(counts["save"]),
            "buy_click_rate": rate(counts["buy"]),
            "top_products": [{"product_id": p, "score": int(s)} for p, s in top.items()]}


def read_secrets(name: str) -> dict | None:
    """The store's secrets if they were set up; never creates them (safe for unauthenticated requests)."""
    if not valid_store_name(name):
        return None
    path = STORES_DIR / name / "secrets.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


def store_secrets(name: str) -> dict:
    """The store's webhook secret and API key (stores/<name>/secrets.json, created on first use, never committed)."""
    check_store_name(name)
    out = STORES_DIR / name
    if not (out / "catalog.pkl").exists():
        raise ValueError(f"no store {name!r}; import it first")
    path = out / "secrets.json"
    with _store_locks[f"{name}:secrets"]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        values = {"webhook_secret": secrets.token_urlsafe(32), "api_key": secrets.token_urlsafe(32)}
        path.write_text(json.dumps(values), encoding="utf-8")
        os.chmod(path, 0o600)                              # owner only (on Linux/macOS)
        return values


def set_webhook_secret(name: str, secret: str) -> None:
    """Save the key the platform signs webhooks with (Shopify shows it on its Webhooks settings page; for
    WooCommerce you may paste ours into the webhook form instead)."""
    secret = str(secret).strip()
    if len(secret) < 16 or any(c.isspace() for c in secret):
        raise ValueError("the signing key should be at least 16 characters, without spaces")
    values = store_secrets(name)
    with _store_locks[f"{name}:secrets"]:
        values["webhook_secret"] = secret
        (STORES_DIR / name / "secrets.json").write_text(json.dumps(values), encoding="utf-8")


def save_store_files(out: Path, catalog: pd.DataFrame, emb: np.ndarray) -> None:
    """Write embeddings, ids and catalog to temporary files, then swap them in (catalog last: its change makes the
    API reload). If a swap fails, the previous files are put back, so the store never mixes two imports."""
    files = {"embeddings.npy": emb.astype(np.float32),
             "product_ids.npy": np.asarray(catalog["product_id"].tolist(), dtype=str)}
    tag = f"{os.getpid()}-{secrets.token_hex(4)}"           # private temp names: never touch another writer's
    for name, arr in files.items():
        with open(out / f"{name}.{tag}.tmp", "wb") as f:
            np.save(f, arr)
    catalog.to_pickle(out / f"catalog.pkl.{tag}.tmp")
    moved = []
    try:
        for name in (*files, "catalog.pkl"):
            if name != "catalog.pkl" and (out / name).exists():
                os.replace(out / name, out / f"{name}.{tag}.bak")
                moved.append(name)
            os.replace(out / f"{name}.{tag}.tmp", out / name)
    except Exception:
        for name in moved:
            os.replace(out / f"{name}.{tag}.bak", out / name)
        raise
    finally:
        for leftover in [*out.glob(f"*.{tag}.tmp"), *out.glob(f"*.{tag}.bak")]:
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
    if not host or is_ip or "." not in host or host.endswith((".local", ".localhost", ".internal")) \
            or not re.fullmatch(r"[a-z0-9.-]+", host):
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
    parser.add_argument("--update-popularity", action="store_true",
                        help="recompute popularity from shopper events (no re-embedding)")
    parser.add_argument("--events-summary", action="store_true", help="print what shoppers did in the last 30 days")
    parser.add_argument("--secrets", action="store_true",
                        help="print the store's webhook paths, webhook secret and API key (creates them once)")
    parser.add_argument("--set-webhook-secret", default="", help="save the signing key Shopify shows for webhooks")
    parser.add_argument("--preset", default="generic", choices=sorted(PRESETS))
    parser.add_argument("--map", default="", help="field=column pairs, e.g. 'title=Name,price=Cost'")
    args = parser.parse_args(argv)
    if args.update_popularity:
        log = update_popularity(args.name)
        print(f"popularity updated for {log['products']} products")
        return
    if args.events_summary:
        print(json.dumps(event_summary(args.name), indent=2))
        return
    if args.set_webhook_secret:
        set_webhook_secret(args.name, args.set_webhook_secret)
        print("saved")
        return
    if args.secrets:
        values = store_secrets(args.name)
        print(json.dumps({"shopify_webhook": f"/webhooks/{args.name}/shopify",
                          "woocommerce_webhook": f"/webhooks/{args.name}/woocommerce", **values}, indent=2))
        return
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
