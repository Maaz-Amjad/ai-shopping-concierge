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
    assert [s.label for s in at.sidebar.selectbox][:2] == ["Store", "Shopper"]
    at.text_input(key="query").set_value("ice maker").run()
    assert not at.exception
    assert "`category: Ice Makers`" in " ".join(m.value for m in at.markdown)
    assert at.info[0].value.startswith("Two \[ice makers\]\(http://x\)")      # link from store/LLM text escaped
    cards = [m.value for m in at.markdown if m.value.startswith("**1.")]
    assert cards and "ice maker a" in cards[0]
    at.sidebar.selectbox[1].select_index(1).run()                                    # returning shopper
    assert not at.exception
    assert any("personalization on" in c.value for c in at.caption)
