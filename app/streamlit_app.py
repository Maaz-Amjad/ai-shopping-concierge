"""Shopping Concierge demo (FR-U2): ask in plain words, see the enforced filters, ranked products and a grounded answer.

Run: .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import re  # noqa: E402
import tempfile  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import load_config  # noqa: E402
from src.pipeline import get_similar, load_resources, result_to_dict, search_products  # noqa: E402
from src.store import PRESETS, STORES_DIR, build_store, list_stores  # noqa: E402

BENCHMARK = "Appliances (benchmark catalog)"
EXAMPLES = ["quiet portable countertop ice maker under $150", "refrigerator water filter, not Samsung",
            "something to stop my dryer from squeaking", "genuine GE brand refrigerator water filter",
            "LG freezer under $50"]

st.set_page_config(page_title="Shopping Concierge", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner="Loading catalog, embeddings and model…")
def load(store: str | None, version: float):
    cfg = load_config()
    return load_resources(cfg, max_calls=cfg.get("app", {}).get("max_llm_calls"), store=store)


def resources(store: str | None):
    """Cached per store and per import (a re-imported store is reloaded)."""
    path = STORES_DIR / store / "catalog.pkl" if store else None
    return load(store, path.stat().st_mtime if path is not None and path.exists() else 0.0)


MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+!|~$-])")


def md(text) -> str:
    """Escape markdown so store text cannot add links, images or formatting."""
    return MD_SPECIAL.sub(lambda m: "\\" + m.group(1), str(text or ""))


@st.cache_data(show_spinner=False)
def shoppers(store: str | None) -> pd.DataFrame:
    """Returning shoppers to pick from: the 8 seeded experiment users (10+ interactions), with recent purchases."""
    if store:
        return pd.DataFrame(columns=["user_id", "label", "recent"])
    from src.data import load_interactions
    from src.personalize import pick_users

    inter = load_interactions()
    titles = resources(None).catalog.set_index("product_id")["title"]
    rows = []
    for i, uid in enumerate(pick_users(inter, n=8, min_interactions=10, seed=42), start=1):
        h = inter[inter["user_id"] == uid].sort_values("timestamp", ascending=False)
        recent = [titles.get(p, p)[:70] for p in h["product_id"].head(3)]
        rows.append({"user_id": uid, "label": f"Returning shopper {i} ({len(h)} purchases)", "recent": recent})
    return pd.DataFrame(rows)


def money(value) -> str:
    return "price not listed" if value is None else f"${value:,.2f}"


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Shopping Concierge")
    stores = [BENCHMARK] + list_stores()
    store_label = st.selectbox("Store", stores, help="Import your own store in the 'Connect your store' tab.")
    store = None if store_label == BENCHMARK else store_label
    res = resources(store)
    people = shoppers(store)
    options = ["New shopper (no history)"] + people["label"].tolist()
    who = st.selectbox("Shopper", options, help="Returning shoppers get results re-ranked toward their history.")
    user_id = None
    if who != options[0]:
        row = people[people["label"] == who].iloc[0]
        user_id = row["user_id"]
        st.caption("Recently bought:\n" + "\n".join(f"- {t}" for t in row["recent"]))
    n = st.slider("Results", 3, 10, 5)
    with st.expander("Optional filters (override the parsed ones)"):
        f_max = st.number_input("Max price ($)", min_value=0.0, value=0.0, step=5.0)
        f_cat = st.selectbox("Category", ["(any)"] + res.meta.categories)
        f_brand = st.selectbox("Brand", ["(any)"] + res.meta.prompt_brands)
    st.caption(f"{len(res.catalog):,} products · {len(res.meta.categories)} categories")

search_tab, connect_tab, about_tab = st.tabs(["Search", "Connect your store", "How it works"])

# ---------------- search ----------------
with search_tab:
    st.subheader("What are you looking for?")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state["query"] = example
    query = st.text_input("Describe it in your own words", key="query",
                          placeholder="e.g. quiet portable countertop ice maker under $150")
    key = (query, user_id, n, store, f_max, f_cat, f_brand)
    if st.button("Search", type="primary") or (query and st.session_state.get("last") != key):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            st.session_state["last"] = key
            with st.spinner("Parsing, filtering, ranking and writing the answer…"):
                out = search_products(query, res, n=n, user_id=user_id,
                                      max_price=f_max or None,
                                      category=None if f_cat == "(any)" else f_cat,
                                      brand=None if f_brand == "(any)" else f_brand)
            st.session_state["result"] = result_to_dict(query, out, res)

    result = st.session_state.get("result")
    if result:
        pq = result["parsed"]
        chips = [f"max ${pq['max_price']:g}" if pq["max_price"] is not None else None,
                 f"min ${pq['min_price']:g}" if pq["min_price"] is not None else None,
                 f"category: {pq['category']}" if pq["category"] else None,
                 f"brand: {pq['brand']}" if pq["brand"] else None]
        chips = [c for c in chips if c]
        st.markdown("**Hard limits enforced:** " + (" · ".join(f"`{c}`" for c in chips) if chips else "none") +
                    f"  \n**Searched for:** _{pq['semantic_query']}_")
        flags = result["flags"]
        notes = [line for line in flags.get("parse_log", []) if line]
        if flags.get("price_excluded"):
            notes.append(f"{flags['price_excluded']} matching products have no listed price and were left out "
                         "because a price limit is active.")
        if notes:
            st.caption(" · ".join(notes))

        st.markdown("#### Answer")
        st.info(md(result["answer"]))
        if flags.get("generation_fallback") == "template":
            st.caption("The LLM answer did not pass the grounding check twice, so this answer was built from catalog "
                       "fields only.")

        if result["products"]:
            st.markdown("#### Ranked products")
            for p in result["products"]:
                with st.container(border=True):
                    left, right = st.columns([4, 2])
                    left.markdown(f"**{p['rank']}. {md(p['title'])}**")
                    left.caption(" · ".join(x for x in [p["brand"], p["category"], money(p["price"])] if x))
                    if p.get("reason"):
                        left.markdown(md(p["reason"]))
                    right.metric("Score", f"{p['final']:.2f}")
                    right.caption(f"relevance {p['s_sem']:.2f} · personal {p['s_pers']:.2f} · "
                                  f"popularity {p['s_pop']:.2f}")
                    with left.expander("Similar products"):
                        sim = get_similar(p["product_id"], res, 3)
                        st.dataframe(sim[["title", "brand", "price", "similarity"]], hide_index=True)
        t = result["timings_ms"]
        st.caption(f"parse {t['parse']:.0f} ms · retrieve {t['retrieve']:.0f} ms · rank {t['rank']:.0f} ms · "
                   f"answer {t['generate']:.0f} ms · personalization {'off (new shopper)' if flags['cold_start'] else 'on'}")

# ---------------- connector ----------------
with connect_tab:
    st.subheader("Plug the concierge into an existing store")
    st.markdown("**1. Import your products** (CSV or JSON export, a file path or a URL). Presets: "
                + ", ".join(f"`{p}`" for p in PRESETS) + ".")
    st.code(".venv/Scripts/python.exe -m src.store --name mystore --file products_export.csv --preset shopify\n"
            ".venv/Scripts/python.exe -m src.store --name mystore --file https://mystore.com/products.json "
            "--preset shopify_json\n"
            ".venv/Scripts/python.exe -m src.store --name mystore --file feed.csv --map title=Name,price=Cost",
            language="bash")
    uploaded = st.file_uploader("…or import a CSV/JSON export here", type=["csv", "json"])
    c1, c2 = st.columns(2)
    new_name = c1.text_input("Store name", value="mystore")
    preset = c2.selectbox("Format", list(PRESETS), index=list(PRESETS).index("shopify"))
    if uploaded and st.button("Import and index"):
        suffix = ".json" if uploaded.name.lower().endswith(".json") else ".csv"   # only the extension is used
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
        path = Path(tmp.name)
        try:
            with st.spinner("Cleaning and embedding products…"):
                log = build_store(new_name, path, preset, model=res.model)
            st.success(f"Imported {log['products']} products into store '{new_name}'. Pick it in the sidebar.")
            st.json(log)
        except ValueError as exc:
            st.error(str(exc))
        finally:
            path.unlink(missing_ok=True)
    st.markdown("**2. Start the API** (read-only; serves search, similar products and the widget):")
    st.code(".venv/Scripts/python.exe -m uvicorn app.api:app --port 8000", language="bash")
    st.markdown("**3. Paste two lines into any page of your store:**")
    st.code('<div id="concierge-widget"></div>\n'
            '<script src="http://localhost:8000/widget.js" data-store="mystore"></script>', language="html")
    st.markdown("**4. Let AI shopping agents use it (MCP).** Add to the MCP config of Claude Desktop or Claude Code:")
    st.code('{"mcpServers": {"shopping-concierge": {\n'
            f'  "command": "{(ROOT / ".venv/Scripts/python.exe").as_posix()}",\n'
            '  "args": ["-m", "app.mcp_server"],\n'
            f'  "cwd": "{ROOT.as_posix()}",\n'
            '  "env": {"CONCIERGE_STORE": "mystore"}}}}', language="json")

# ---------------- about ----------------
with about_tab:
    st.markdown("""
1. **Understand** – Gemini turns your words into hard limits (price, category, brand) plus a descriptive query.
   Category and brand must exist in the catalog; anything else is dropped and shown.
2. **Enforce** – Python keeps only products that satisfy every limit (the LLM never filters or ranks).
3. **Retrieve** – BGE embeddings find the products closest in meaning among the allowed ones.
4. **Rank** – 0.7 × relevance + 0.25 × personal fit (returning shoppers) + 0.05 × popularity.
5. **Answer** – Gemini writes a short answer from catalog fields only; a validator rejects any product, price or
   brand that is not in the results, and a template answer is used if it fails twice.

Measured on 15 held-out test queries: every recommended product obeyed every stated limit (100 % vs 43–63 % for plain
search), query parsing F1 0.94, and 22/22 answers passed the grounding check. Details: `docs/final_report.md`.
""")
