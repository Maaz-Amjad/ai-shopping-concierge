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
