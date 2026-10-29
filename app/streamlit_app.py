"""Shopping Assistant demo (FR-U2): shop in plain words, add your own store, connect a website or an AI assistant.

Everything is done with buttons; no commands. Run: .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import connection  # noqa: E402
from app.connection import add_to_claude_desktop, claude_desktop_config_path, claude_desktop_entry  # noqa: E402
from src import load_config  # noqa: E402
from src.agent import chat_turn, run_action  # noqa: E402
from src.pipeline import app_config, get_similar, load_resources, result_to_dict, search_products  # noqa: E402
from src.store import (FIELDS, OPTIONAL_FIELDS, STORES_DIR, build_store_from_feed, detect_preset, link_template,  # noqa: E402
                       list_stores, read_feed, shopify_feed_url, store_name_from)

DEMO = "Demo appliance store"
EXAMPLES = ["quiet portable countertop ice maker under $150", "refrigerator water filter, not Samsung",
            "something to stop my dryer from squeaking", "genuine GE brand refrigerator water filter",
            "LG freezer under $50"]
FORMAT_NAMES = {"shopify": "Shopify product export", "shopify_json": "Shopify store data",
                "woocommerce": "WooCommerce product export", "generic": "standard product list"}
FIELD_QUESTIONS = {"product_id": "Which column is the product's code or ID?", "title": "Which column is the product name?",
                   "price": "Which column is the price?", "brand": "Which column is the brand?",
                   "category": "Which column is the category?", "description": "Which column describes the product?",
                   "features": "Which column lists features or tags?",
                   "rating": "Which column is the average star rating (optional)?",
                   "rating_count": "Which column is the number of ratings or reviews (optional)?"}
NONE = "— not in my file —"

st.set_page_config(page_title="Shopping Assistant", page_icon="🛒", layout="wide")


# ---------------- helpers ----------------
@st.cache_resource(show_spinner="Getting the store ready (first time takes a minute)…")
def load(store: str | None, version: float):
    cfg = load_config()
    return load_resources(app_config(cfg), max_calls=cfg.get("app", {}).get("max_llm_calls"), store=store)


def resources(store: str | None):
    """Cached per store and per import (a re-imported store is reloaded)."""
    path = STORES_DIR / store / "catalog.pkl" if store else None
    return load(store, path.stat().st_mtime if path is not None and path.exists() else 0.0)


@st.cache_resource
def connection_state() -> dict:
    """The website connection started from this app (shared across browser refreshes)."""
    return {"proc": None}


MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+!|~$:.-])")    # ':' and '.' stop bare links being auto-linked


def md(text) -> str:
    """Escape markdown so store text cannot add links, images or formatting."""
    return MD_SPECIAL.sub(lambda m: "\\" + m.group(1), str(text or ""))


@st.cache_data(show_spinner=False)
def shoppers(store: str | None) -> pd.DataFrame:
    """Returning shoppers to pick from: the 8 seeded experiment users (10+ purchases), with recent purchases."""
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
        rows.append({"user_id": uid, "label": f"Returning shopper {i} ({len(h)} past purchases)", "recent": recent})
    return pd.DataFrame(rows)


def chip(text: str) -> str:
    """Inline code chip; text inside backticks is shown literally, so only a backtick could break out."""
    return "`" + str(text).replace("`", "'") + "`"


def money(value) -> str:
    return "price not listed" if value is None else f"${value:,.2f}"


def stars(p: dict) -> str | None:
    """'★ 4.4 (327 ratings)' when the store has ratings for the product."""
    avg = p.get("avg_rating")
    return None if avg is None else f"★ {avg:.1f} ({int(p.get('rating_count') or 0):,} ratings)"


def understood(pq: dict) -> list[str]:
    out = []
    if pq["min_price"] is not None and pq["max_price"] is not None:
        out.append(f"between ${pq['min_price']:g} and ${pq['max_price']:g}")
    elif pq["max_price"] is not None:
        out.append(f"under ${pq['max_price']:g}")
    elif pq["min_price"] is not None:
        out.append(f"over ${pq['min_price']:g}")
    if pq["category"]:
        out.append(f"category: {pq['category']}")
    if pq["brand"]:
        out.append(f"brand: {pq['brand']}")
    out += [f"not {t}" for t in pq.get("exclude_terms", [])]
    return out


# a store chosen with "Start shopping in this store" is applied before the sidebar is drawn
if "pending_store" in st.session_state:
    st.session_state["store_choice"] = st.session_state.pop("pending_store")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("🛒 Shopping Assistant")
    shops = [DEMO] + list_stores()
    if st.session_state.get("store_choice") not in shops:
        st.session_state["store_choice"] = DEMO
    shop = st.selectbox("Shop", shops, key="store_choice", help="Add your own store in the 'Add your store' tab.")
    store = None if shop == DEMO else shop
    res = resources(store)
    people = shoppers(store)
    options = ["New visitor"] + people["label"].tolist()
    who = st.selectbox("Who is shopping?", options,
                       help="Returning shoppers see results that also fit what they bought before.")
    user_id = None
    if who != options[0]:
        row = people[people["label"] == who].iloc[0]
        user_id = row["user_id"]
        st.caption("Recently bought:\n" + "\n".join(f"- {md(t)}" for t in row["recent"]))
    elif store:
        st.caption("Your store has no purchase history here yet, so everyone shops as a new visitor.")
    n = st.slider("How many suggestions?", 3, 10, 5)
    with st.expander("Narrow it down (optional)"):
        f_max = st.number_input("Highest price ($, 0 = no limit)", min_value=0.0, value=0.0, step=5.0)
        f_cat = st.selectbox("Category", ["Any"] + res.meta.categories)
        f_brand = st.selectbox("Brand", ["Any"] + res.meta.prompt_brands)
    st.caption(f"{len(res.catalog):,} products · {len(res.meta.categories)} categories")

chat_tab, shop_tab, add_tab, about_tab = st.tabs(["💬 Chat", "🔎 Shop", "🏪 Add your store", "ℹ️ How it works"])

# ---------------- chat ----------------
with chat_tab:
    chat_key = (store, user_id)
    if st.session_state.get("chat_key") != chat_key:          # new shop or shopper: new conversation
        st.session_state.update(chat_key=chat_key, chat_state=None, chat_log=[])
    link = link_template(store)

    def remember(out) -> None:
        st.session_state["chat_state"] = out.state.model_dump()
        st.session_state["chat_log"].append({"role": "assistant", "text": out.text, "products": out.products,
                                             "comparison": out.comparison, "link": out.link})

    top_left, top_right = st.columns([5, 1])
    top_left.caption("Chat with the shopping assistant: ask for products, then say things like “cheaper”, "
                     "“only GE”, “compare 1 and 2”, “save the first one” or “I'll take number 2”.")
    if top_right.button("Start over"):
        st.session_state.update(chat_state=None, chat_log=[])
        st.rerun()
    if not st.session_state["chat_log"]:
        with st.chat_message("assistant"):
            st.markdown("Hi! What are you shopping for today?")
    for i, entry in enumerate(st.session_state["chat_log"]):
        with st.chat_message(entry["role"]):
            st.markdown(md(entry["text"]))
            if entry.get("comparison"):
                st.dataframe(pd.DataFrame(entry["comparison"]), hide_index=True)
            if entry.get("link"):
                st.link_button("Open the product page", entry["link"])
            for p in entry.get("products") or []:
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([6, 1, 1, 1])
                    c1.markdown(f"**{p['number']}. {md(p['title'])}**  \n"
                                + " · ".join(md(x) for x in [p.get("brand"), money(p.get("price")), stars(p)] if x))
                    if p.get("reason"):
                        c1.caption(md(p["reason"]))
                    last = i == len(st.session_state["chat_log"]) - 1          # buttons act on the latest list
                    for col, label, name in ((c2, "Save", "save"), (c3, "Similar", "similar"), (c4, "Buy", "buy")):
                        args = {"product_id": p["product_id"]}                  # this card's product
                        if col.button(label, key=f"{name}-{i}-{p['number']}", disabled=not last):
                            remember(run_action(name, args, st.session_state["chat_state"], res, n=n, link=link))
                            st.rerun()
    shortlist = (st.session_state.get("chat_state") or {}).get("shortlist") or []
    if shortlist:
        with st.expander(f"Your shortlist ({len(shortlist)})"):
            for j, pid in enumerate(shortlist, start=1):
                row = res.catalog[res.catalog["product_id"] == pid].iloc[0]
                st.markdown(f"{j}. {md(str(row['title'])[:90])} — {money(None if pd.isna(row['price']) else row['price'])}")
    message = st.chat_input("Message the shopping assistant")
    if message:
        st.session_state["chat_log"].append({"role": "user", "text": message})
        with st.spinner("Thinking…"):
            remember(chat_turn(message, st.session_state["chat_state"], res, user_id=user_id, n=n, link=link))
        st.rerun()

# ---------------- shop ----------------
with shop_tab:
    st.subheader("What are you looking for?")
    st.caption("Describe it the way you would to a shop assistant: what it is, what matters to you, your budget.")
    st.write("Try one:")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state["query"] = example
    query = st.text_input("Your request", key="query", label_visibility="collapsed",
                          placeholder="e.g. quiet portable countertop ice maker under $150")
    if st.session_state.get("result", {}).get("store", store) != store:
        st.session_state.pop("result", None)            # never show another shop's results
    key = (query, user_id, store)                       # a new request, shopper or shop searches right away;
    st.caption("Changed the number of suggestions or the optional limits? Press Search to update.")
    if st.button("Search", type="primary") or (query and st.session_state.get("last") != key):
        if not query.strip():
            st.warning("Please type what you are looking for.")
        else:
            st.session_state["last"] = key
            with st.spinner("Finding the best matches and writing a short answer…"):
                out = search_products(query, res, n=n, user_id=user_id, max_price=f_max or None,
                                      category=None if f_cat == "Any" else f_cat,
                                      brand=None if f_brand == "Any" else f_brand)
            st.session_state["result"] = {**result_to_dict(query, out, res), "store": store}

    result = st.session_state.get("result")
    if result:
        pq, flags = result["parsed"], result["flags"]
        limits = understood(pq)
        st.markdown("**I understood:** " + (" · ".join(chip(c) for c in limits) if limits
                                             else "no price, brand or category limits") +
                    f"  \n**Looking for:** _{md(pq['semantic_query'])}_")
        if flags.get("price_excluded"):
            st.caption(f"{flags['price_excluded']} matching products have no price listed, so they are not shown "
                       "while a price limit is set.")

        st.info(md(result["answer"]))

        if result["products"]:
            for p in result["products"]:
                with st.container(border=True):
                    left, right = st.columns([5, 2])
                    left.markdown(f"**{p['rank']}. {md(p['title'])}**")
                    left.caption(" · ".join(md(x) for x in [p["brand"], p["category"], money(p["price"]), stars(p)]
                                            if x))
                    if p.get("reason"):
                        left.markdown(md(p["reason"]))
                    right.metric("Match", f"{round(100 * p['final'])}%")
                    with right.popover("Why this order?"):
                        st.write("Fits your request")
                        st.progress(min(max(p["s_sem"], 0.0), 1.0))
                        st.write("Fits your past purchases")
                        st.progress(min(max(p["s_pers"], 0.0), 1.0))
                        st.write("Popular with shoppers")
                        st.progress(min(max(p["s_pop"], 0.0), 1.0))
                        if p.get("avg_rating") is not None:
                            st.write("Well rated by buyers")
                            st.progress(min(max(p.get("s_rat") or 0.0, 0.0), 1.0))
                    with left.expander("Similar products"):
                        for _, s in get_similar(p["product_id"], res, 3).iterrows():
                            st.markdown(f"- {md(str(s['title'])[:90])} — "
                                        f"{money(None if pd.isna(s['price']) else s['price'])}")
        with st.expander("Details for the curious"):
            t = result["timings_ms"]
            st.caption(f"Understanding {t['parse']:.0f} ms · searching {t['retrieve']:.0f} ms · ranking "
                       f"{t['rank']:.0f} ms · writing the answer {t['generate']:.0f} ms · "
                       f"Personalization: {'off (new visitor)' if flags['cold_start'] else 'on'}")
            notes = [line for line in flags.get("parse_log", []) if line]
            if notes:
                st.caption("Notes: " + " · ".join(md(x) for x in notes))
            if flags.get("generation_fallback") == "template":
                st.caption("The written answer failed the fact check twice, so this answer uses catalog facts only.")

# ---------------- add your store ----------------
with add_tab:
    st.subheader("Use the assistant with your own products")

    with st.container(border=True):
        st.markdown("### Step 1 · Bring in your products")
        label = st.text_input("Name your store", value="My store")
        name = store_name_from(label)
        how = st.radio("How would you like to add them?", ["Upload a product file", "Link my Shopify store"],
                       horizontal=True)
        feed, preset, mapping, source = None, None, None, "upload"
        if how == "Upload a product file":
            st.caption("Export your products from Shopify (Products → Export), WooCommerce (Products → Export) or "
                       "any other system as a CSV or JSON file, then drop it here.")
            uploaded = st.file_uploader("Product file", type=["csv", "json"], label_visibility="collapsed")
            if uploaded:
                suffix = ".json" if uploaded.name.lower().endswith(".json") else ".csv"   # only the extension is used
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                try:
                    feed = read_feed(tmp.name)
                except Exception:  # noqa: BLE001 - any unreadable file gets the same friendly message
                    st.error("This file could not be read. Please upload a CSV or JSON product export.")
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
            if feed is not None:
                preset = detect_preset(feed)
                if preset:
                    st.success(f"Recognised a {FORMAT_NAMES[preset]} with {len(feed):,} rows.")
                else:
                    st.info("I don't recognise this layout yet. Tell me which column is which:")
                    columns = [NONE] + [str(c) for c in feed.columns]
                    mapping = {}
                    for field in FIELDS + list(OPTIONAL_FIELDS):
                        choice = st.selectbox(FIELD_QUESTIONS[field], columns, key=f"map_{field}")
                        if choice != NONE:
                            mapping[field] = choice
                    preset = "generic"
                with st.expander("Preview your file"):
                    st.dataframe(feed.head(5), hide_index=True)
        else:
            address = st.text_input("Your Shopify store's web address", placeholder="mystore.com")
            st.caption("Only products that are visible on your public store are read; nothing is changed.")
            if address and st.button("Read my store"):
                try:
                    with st.spinner("Reading your store's products…"):
                        st.session_state["shopify_feed"] = read_feed(shopify_feed_url(address))
                        st.session_state["shopify_source"] = address
                except Exception:  # noqa: BLE001 - network or format problems get one friendly message
                    st.error("Couldn't read that store. Check the address, and that it is a Shopify store.")
            if st.session_state.get("shopify_feed") is not None:
                feed, preset = st.session_state["shopify_feed"], "shopify_json"
                source = st.session_state.get("shopify_source", "shopify")
                st.success(f"Found {len(feed):,} products.")
        if feed is not None:
            ready = mapping is None or {"product_id", "title"} <= set(mapping)   # None = format recognised
            if not ready:
                st.warning("Please choose at least the product code/ID and the product name columns.")
            if st.button(f"Add my products as “{name}”", type="primary", disabled=not ready):
                try:
                    with st.spinner("Reading and indexing your products (about a minute per few thousand)…"):
                        log = build_store_from_feed(name, feed, preset, mapping, model=res.model, source=source)
                    st.session_state["added"] = log
                except ValueError as exc:
                    st.error(f"Couldn't add the products: {exc}")
        log = st.session_state.get("added")
        if log:
            st.success(f"Done! {log['products']:,} products are ready in “{log['store']}” "
                       f"({log['n_categories']} categories; {log['price_missing']} without a price).")
            if st.button("Start shopping in this store"):
                st.session_state["pending_store"] = log["store"]
                st.session_state.pop("added", None)
                st.rerun()

    with st.container(border=True):
        st.markdown("### Step 2 · Put the assistant on your website")
        target = store or ""
        st.caption(f"Uses the shop selected on the left: **{md(shop)}**.")
        site = st.text_input("Your website's address (so it is allowed to use the assistant)",
                             placeholder="mystore.com")
        state = connection_state()
        running = connection.is_running()
        if running:
            st.markdown("🟢 **The connection is on.**")
            c1, c2 = st.columns(2)
            c1.link_button("See it on a sample web page", f"{connection.URL}/" + (f"?store={target}" if target else ""))
            if connection.started_by_app():
                if c2.button("Turn the connection off"):
                    if state["proc"] is not None:
                        connection.stop(state["proc"])
                    else:
                        connection.stop_previous()      # started before the app was restarted
                    state["proc"] = None
                    st.rerun()
                st.caption("To change the website address, turn the connection off and on again.")
            else:
                st.caption("The connection was started outside this app; stop it where it was started.")
        else:
            st.markdown("⚪ **The connection is off.**")
            if st.button("Turn the connection on", type="primary"):
                try:
                    with st.spinner("Starting (this can take up to a minute)…"):
                        state["proc"] = connection.start([connection.website_origin(site)] if site else [])
                    st.rerun()
                except RuntimeError as exc:
                    st.error(f"Couldn't start the connection: {exc}")
        with st.expander("Add the search box to your website"):
            st.markdown("Click the copy icon on the box below, then paste it where the search box should appear:")
            st.code(f'<div id="concierge-widget"></div>\n'
                    f'<script src="{connection.URL}/widget.js" data-store="{target}"></script>', language="html")
            st.markdown("- **Shopify:** Online Store → Themes → Customize → add a *Custom Liquid* section → paste.\n"
                        "- **WooCommerce / WordPress:** edit a page → add a *Custom HTML* block → paste.\n"
                        "- **Other websites:** paste it into the page's HTML, or send it to your web developer.")
            st.caption("This computer runs the assistant. For customers on the internet to use it, it has to run on a "
                       "computer they can reach (your web host or developer can set this up).")

    with st.container(border=True):
        st.markdown("### Step 3 · Let AI assistants shop your store")
        st.caption("AI assistants such as Claude Desktop can then search your products and suggest them, using only "
                   "your real product data. They can only read; they cannot change anything.")
        cfg_path = claude_desktop_config_path()
        if st.button("Connect to Claude Desktop on this computer"):
            try:
                backup = add_to_claude_desktop(store, cfg_path)
                st.success("Done. Quit and reopen Claude Desktop, then ask it to find products in your store."
                           + (" (Your previous settings were backed up.)" if backup else ""))
            except (OSError, ValueError) as exc:
                st.error(f"Couldn't update Claude Desktop's settings: {exc}")
        st.caption("Prefer to do it by hand, or using another AI assistant? Download the settings and give them to "
                   "whoever manages that assistant. Don't replace your existing settings file with this one.")
        st.download_button("Download the settings", file_name="shopping-assistant-settings.json",
                            mime="application/json",
                            data=json.dumps(claude_desktop_entry(store), indent=2))

# ---------------- how it works ----------------
with about_tab:
    st.markdown("""
**What happens when you search**

1. **Understanding your request** – an AI model reads your words and picks out firm limits such as a budget, a brand or
   a product type. Everything else ("quiet", "for a small kitchen") is kept as a description.
2. **Keeping to your limits** – only products that meet every limit are considered. This step is plain, exact code;
   the AI cannot override it.
3. **Finding the best matches** – products are compared with your description by meaning, not just by words.
4. **Putting them in order** – mostly by how well they fit your request; for returning shoppers, a little by what
   they bought before; and a little by popularity.
5. **Writing a short answer** – the AI writes a few sentences using only the facts of the products shown. A checker
   rejects any product, price or brand that isn't really there.

**How well it works**, on 15 test requests that were never used to tune it:

- Every suggested product kept to the stated limits (on the 8 test requests that had limits and matching products;
  plain search without step 2 kept to them only 43–63 % of the time).
- Picking out the limits scored 0.94 on a 0–1 accuracy scale (F1).

Across all 25 benchmark requests, all 22 written answers passed the fact check (the other 3 had no matching products
and got a plain "nothing matches" answer). Full results: `docs/final_report.md`.
""")
