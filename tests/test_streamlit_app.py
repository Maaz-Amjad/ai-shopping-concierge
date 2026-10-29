"""Streamlit demo, headless (AppTest) on the toy catalog with a fake LLM: search, shopper switch, escaping."""

import json

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from test_pipeline import gen, resources


def test_demo_app_searches_personalizes_and_escapes_store_text(tmp_path, monkeypatch):
    from src import data, pipeline, store

    parsed = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    res = resources(tmp_path, [parsed, gen(["A", "B"], "Two [ice makers](http://x)."), gen(["A", "B"])])
    monkeypatch.setattr(pipeline, "load_resources", lambda *a, **k: res)
    monkeypatch.setattr(store, "list_stores", lambda: [])
    history = pd.DataFrame({"user_id": ["u_lg"] * 10, "product_id": ["B"] * 10, "rating": 5.0,
                            "timestamp": range(10)})
    monkeypatch.setattr(data, "load_interactions", lambda: history)
    st.cache_resource.clear()
    st.cache_data.clear()

    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
    assert not at.exception
    assert [s.label for s in at.sidebar.selectbox][:2] == ["Shop", "Who is shopping?"]
    assert not any("python" in c.value or "uvicorn" in c.value for c in at.code if "widget" not in c.value)
    at.text_input(key="query").set_value("ice maker").run()
    assert not at.exception
    assert "`category: Ice Makers`" in " ".join(m.value for m in at.markdown)       # "I understood: ..."
    assert at.info[0].value.startswith(r"Two \[ice makers\]\(http\://x\)")    # links and bare URLs escaped
    cards = [m.value for m in at.markdown if m.value.startswith("**1.")]
    assert cards and "ice maker a" in cards[0]
    at.sidebar.selectbox[1].select_index(1).run()                                    # returning shopper
    assert not at.exception
    assert any("Personalization: on" in c.value for c in at.caption)


def test_chat_tab_searches_refines_and_saves(tmp_path, monkeypatch):
    from src import data, pipeline, store
    from test_agent import agent_res

    res = agent_res(tmp_path, [{"name": "search", "args": {"request": "ice maker", "category": "Ice Makers"}},
                               {"name": "compare", "args": {"items": [1, 2]}}], [gen(["A", "B"])])
    monkeypatch.setattr(pipeline, "load_resources", lambda *a, **k: res)
    monkeypatch.setattr(store, "list_stores", lambda: [])
    monkeypatch.setattr(data, "load_interactions", lambda: pd.DataFrame(
        {"user_id": ["u_lg"] * 10, "product_id": ["B"] * 10, "rating": 5.0, "timestamp": range(10)}))
    st.cache_resource.clear()
    st.cache_data.clear()

    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
    assert at.tabs[0].label == "💬 Chat" and not at.exception
    at.chat_input[0].set_value("I need an ice maker").run()
    assert not at.exception
    assert any(m.value.startswith("**1. ice maker a**") for m in at.markdown)
    at.button(key="save-1-2").click().run()                                    # Save on card 2, no LLM call
    assert not at.exception and "You have 1 saved" in " ".join(m.value for m in at.markdown)
    at.chat_input[0].set_value("compare 1 and 2").run()
    assert not at.exception and at.dataframe                                    # comparison table shown


def test_chat_buy_on_a_shortlist_card_opens_that_product(tmp_path, monkeypatch):
    from src import data, pipeline, store
    from test_agent import agent_res

    res = agent_res(tmp_path, [{"name": "search", "args": {"request": "ice maker", "category": "Ice Makers"}},
                               {"name": "show_shortlist", "args": {}}], [gen(["A", "B"])])
    monkeypatch.setattr(pipeline, "load_resources", lambda *a, **k: res)
    monkeypatch.setattr(store, "list_stores", lambda: [])
    monkeypatch.setattr(data, "load_interactions", lambda: pd.DataFrame(
        {"user_id": ["u_lg"] * 10, "product_id": ["B"] * 10, "rating": 5.0, "timestamp": range(10)}))
    st.cache_resource.clear()
    st.cache_data.clear()

    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
    at.chat_input[0].set_value("I need an ice maker").run()
    at.button(key="save-1-2").click().run()                                    # saves B
    at.chat_input[0].set_value("what's on my list?").run()                     # shortlist card 1 = B
    at.button(key="buy-4-1").click().run()
    assert not at.exception
    assert at.session_state["chat_log"][-1]["link"].endswith("/B")             # not shown number 1 (A)


def test_chat_cards_show_star_ratings(tmp_path, monkeypatch):
    import numpy as np

    from src import data, pipeline, store
    from test_agent import agent_res

    res = agent_res(tmp_path, [{"name": "search", "args": {"request": "ice maker", "category": "Ice Makers"}}],
                    [gen(["A", "B"])])
    res.catalog = res.catalog.assign(avg_rating=[4.46, np.nan, 3.0], rating_count=[1234, 0, 4])
    monkeypatch.setattr(pipeline, "load_resources", lambda *a, **k: res)
    monkeypatch.setattr(store, "list_stores", lambda: [])
    monkeypatch.setattr(data, "load_interactions", lambda: pd.DataFrame(
        {"user_id": ["u_lg"] * 10, "product_id": ["B"] * 10, "rating": 5.0, "timestamp": range(10)}))
    st.cache_resource.clear()
    st.cache_data.clear()

    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
    at.chat_input[0].set_value("I need an ice maker").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown).replace("\\", "")              # card text is markdown-escaped
    assert "★ 4.5 (1,234 ratings)" in text and text.count("★") == 1                  # unrated B shows no stars


def test_update_products_button_refreshes_the_selected_store(tmp_path, monkeypatch):
    from src import data, pipeline, store

    res = resources(tmp_path, [])
    (tmp_path / "stores" / "shop1").mkdir(parents=True)
    (tmp_path / "stores" / "shop1" / "import_log.json").write_text(json.dumps({"products": 3, "updated_at": "earlier"}))
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    monkeypatch.setattr(pipeline, "load_resources", lambda *a, **k: res)
    monkeypatch.setattr(store, "list_stores", lambda: ["shop1"])
    monkeypatch.setattr(store, "refresh_store", lambda name, model=None: {
        "products": 4, "added": 1, "removed": 0, "changed": 2, "out_of_stock": 1})
    monkeypatch.setattr(data, "load_interactions", lambda: pd.DataFrame(
        {"user_id": [], "product_id": [], "rating": [], "timestamp": []}))
    st.cache_resource.clear()
    st.cache_data.clear()

    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
    at.sidebar.selectbox[0].select("shop1").run()
    assert not at.exception
    next(b for b in at.button if b.label == "Update products now").click().run()
    assert not at.exception
    assert any("1 new" in s.value and "1 out of stock" in s.value for s in at.success)



def test_instant_updates_section_shows_webhook_urls(tmp_path, monkeypatch):
    from src import data, pipeline, store

    res = resources(tmp_path, [])
    (tmp_path / "stores" / "shop1").mkdir(parents=True)
    (tmp_path / "stores" / "shop1" / "import_log.json").write_text(json.dumps({"products": 3}))
    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    monkeypatch.setattr(pipeline, "load_resources", lambda *a, **k: res)
    monkeypatch.setattr(store, "list_stores", lambda: ["shop1"])
    monkeypatch.setattr(store, "store_secrets", lambda name: {"webhook_secret": "s" * 40, "api_key": "k" * 40})
    monkeypatch.setattr(data, "load_interactions", lambda: pd.DataFrame(
        {"user_id": [], "product_id": [], "rating": [], "timestamp": []}))
    st.cache_resource.clear()
    st.cache_data.clear()

    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
    at.sidebar.selectbox[0].select("shop1").run()
    at.text_input(key="public_api").set_value("https://assistant.example.com/").run()
    assert not at.exception
    codes = [c.value for c in at.code]
    assert "https://assistant.example.com/webhooks/shop1/shopify" in codes and "s" * 40 in codes
