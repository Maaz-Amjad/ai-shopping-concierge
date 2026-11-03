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


# ---- Phase 14: hosted demo (CONCIERGE_DEMO=1) ---------------------------------------------------------------

def demo_app(tmp_path, monkeypatch, parse_and_answers=(), tool_calls=()):
    """Demo-mode app on the toy catalog. Real LLMClients are built by the app, so the provider transports are faked
    (recording the key they were given); the LLM cache and log go to tmp_path."""
    from src import data, llm, pipeline, store

    res = resources(tmp_path, [])
    res.cfg = {**res.cfg, "llm": {"model": "m", "cache_path": str(tmp_path / "cache.jsonl"),
                                  "call_log_path": str(tmp_path / "calls.jsonl"), "rpm": 1e9, "temperature": 0,
                                  "max_output_tokens": 400}}
    calls, answers, tools = [], iter(parse_and_answers), iter(tool_calls)

    def fake_transports_for(provider="gemini", base_url=None, api_key_env=None, api_key=None):
        def transport(*a):
            calls.append((provider, api_key))
            return next(answers)

        def tool_transport(*a):
            calls.append((provider, api_key))
            return json.dumps(next(tools))
        return transport, tool_transport

    monkeypatch.setenv("CONCIERGE_DEMO", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "owner-key")
    monkeypatch.setattr(llm, "transports_for", fake_transports_for)
    monkeypatch.setattr(pipeline, "load_resources", lambda *a, **k: res)
    monkeypatch.setattr(store, "list_stores", lambda: (_ for _ in ()).throw(AssertionError("stores listed")))
    monkeypatch.setattr(data, "load_interactions", lambda: pd.DataFrame(
        {"user_id": ["u_lg"] * 10, "product_id": ["B"] * 10, "rating": 5.0, "timestamp": range(10)}))
    st.cache_resource.clear()
    st.cache_data.clear()
    return AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run(), calls


def test_demo_mode_hides_the_store_owner_sections(tmp_path, monkeypatch):
    at, _ = demo_app(tmp_path, monkeypatch)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["💬 Chat", "🔎 Shop", "ℹ️ How it works"]
    assert "Shop" not in [s.label for s in at.sidebar.selectbox]                    # demo catalog only
    labels = [b.label for b in at.button]
    assert not any("Claude Desktop" in b or "connection" in b or "Update products" in b for b in labels)
    assert not at.code                                                               # no widget code, no secrets


def test_demo_without_a_key_makes_no_llm_call(tmp_path, monkeypatch):
    at, calls = demo_app(tmp_path, monkeypatch)
    at.text_input(key="query").set_value("ice maker").run()
    assert not at.exception and any(m.value.startswith("**1.") for m in at.markdown)   # results still shown
    at.chat_input[0].set_value("I need an ice maker").run()
    assert not at.exception and any(m.value.startswith("**1.") for m in at.markdown)
    assert calls == []                                                               # owner key never used


def test_demo_uses_the_visitors_key_for_their_session_and_never_shows_it(tmp_path, monkeypatch):
    parsed = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    at, calls = demo_app(tmp_path, monkeypatch, [parsed, gen(["A", "B"])])
    at.text_input(key="visitor_key").set_value("visitor-key-123").run()
    at.text_input(key="query").set_value("ice maker").run()
    assert not at.exception
    assert calls and set(calls) == {("gemini", "visitor-key-123")}
    assert "`category: Ice Makers`" in " ".join(m.value for m in at.markdown)        # the LLM parse was used
    shown = " ".join(e.value for e in [*at.markdown, *at.caption, *at.info, *at.code] if isinstance(e.value, str))
    assert "visitor-key-123" not in shown
    for name in ("cache.jsonl", "calls.jsonl"):
        assert "visitor-key-123" not in (tmp_path / name).read_text(encoding="utf-8")


def test_demo_basic_mode_does_not_claim_a_failed_fact_check(tmp_path, monkeypatch):
    at, calls = demo_app(tmp_path, monkeypatch)
    at.text_input(key="query").set_value("ice maker").run()
    captions = " ".join(c.value for c in at.caption)
    assert not at.exception and calls == []
    assert "failed the fact check" not in captions and "basic mode" in captions


def test_demo_key_is_not_sent_to_another_provider_after_switching(tmp_path, monkeypatch):
    at, calls = demo_app(tmp_path, monkeypatch)
    at.text_input(key="visitor_key").set_value("AIza-google-key").run()           # typed for Gemini
    at.selectbox(key="visitor_provider").select("Anthropic (Claude)").run()       # then the provider is changed
    at.text_input(key="query").set_value("ice maker").run()
    assert not at.exception
    assert calls == []                                       # Claude has no key of its own yet: basic mode, no call


def test_demo_basic_mode_applies_the_narrow_it_down_limits_without_llm(tmp_path, monkeypatch):
    at, calls = demo_app(tmp_path, monkeypatch)
    at.sidebar.number_input[0].set_value(110.0).run()                              # highest price $110
    at.text_input(key="query").set_value("ice maker").run()
    cards = [m.value for m in at.markdown if m.value.startswith("**")]
    assert not at.exception and calls == []
    assert any("ice maker a" in c for c in cards) and not any("ice maker b" in c for c in cards)   # B costs $120


# ---- Shop tab UX (3 Nov) --------------------------------------------------------------------------------------

def shop_app(tmp_path, monkeypatch, responses):
    from src import data, pipeline, store

    res = resources(tmp_path, responses)
    monkeypatch.setattr(pipeline, "load_resources", lambda *a, **k: res)
    monkeypatch.setattr(store, "list_stores", lambda: [])
    monkeypatch.setattr(data, "load_interactions", lambda: pd.DataFrame(
        {"user_id": ["u_lg"] * 10, "product_id": ["B"] * 10, "rating": 5.0, "timestamp": range(10)}))
    st.cache_resource.clear()
    st.cache_data.clear()
    return AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()


def cards_shown(at) -> list[str]:
    return [m.value for m in at.markdown if m.value.startswith("**1.") or m.value.startswith("**2.")]


def test_clicking_a_try_one_example_searches_right_away(tmp_path, monkeypatch):
    parsed = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    at = shop_app(tmp_path, monkeypatch, [parsed, gen(["A", "B"])])
    next(b for b in at.button if b.label.startswith("quiet portable countertop")).click().run()
    assert not at.exception
    assert at.text_input(key="query").value.startswith("quiet portable countertop")
    assert cards_shown(at)                                   # results without pressing Search


def test_clearing_the_request_clears_the_results(tmp_path, monkeypatch):
    parsed = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    at = shop_app(tmp_path, monkeypatch, [parsed, gen(["A", "B"])])
    at.text_input(key="query").set_value("ice maker").run()
    assert cards_shown(at)
    at.text_input(key="query").set_value("").run()
    assert not at.exception
    assert not cards_shown(at) and not at.info                # old results and answer are gone
    assert not at.warning                                     # clearing is not an error


def test_recent_searches_are_listed_under_the_box_and_search_again(tmp_path, monkeypatch):
    p1 = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    p2 = json.dumps({"semantic_query": "range hood"})
    at = shop_app(tmp_path, monkeypatch, [p1, gen(["A", "B"]), p2, gen(["C"], "One hood.")])
    at.text_input(key="query").set_value("ice maker").run()
    at.text_input(key="query").set_value("range hood").run()
    recent = [b for b in at.button if b.key and b.key.startswith("recent-")]
    assert [b.label for b in recent] == ["range hood", "ice maker"]               # newest first, this session only
    next(b for b in recent if b.label == "ice maker").click().run()               # answered from the LLM cache
    assert not at.exception
    assert at.text_input(key="query").value == "ice maker" and cards_shown(at)
    assert [b.label for b in at.button if b.key and b.key.startswith("recent-")] == ["ice maker", "range hood"]


def test_a_recent_search_can_be_removed_with_its_cross(tmp_path, monkeypatch):
    p1 = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    p2 = json.dumps({"semantic_query": "range hood"})
    at = shop_app(tmp_path, monkeypatch, [p1, gen(["A", "B"]), p2, gen(["C"], "One hood.")])
    at.text_input(key="query").set_value("ice maker").run()
    at.text_input(key="query").set_value("range hood").run()
    labels = lambda: [b.label for b in at.button if b.key and b.key.startswith("recent-")]   # noqa: E731
    assert labels() == ["range hood", "ice maker"]
    at.button(key="forget-1").click().run()                                       # the cross next to "ice maker"
    assert not at.exception
    assert labels() == ["range hood"]
    assert at.text_input(key="query").value == "range hood" and cards_shown(at)   # no new search, results kept


def test_active_limits_are_shown_and_can_be_cleared(tmp_path, monkeypatch):
    at = shop_app(tmp_path, monkeypatch, [])
    assert "clear_limits" not in [b.key for b in at.button]                         # nothing set: no limits row
    at.sidebar.number_input(key="f_max").set_value(110.0).run()
    assert any("`under $110`" in c.value for c in at.sidebar.caption)
    at.button(key="clear_limits").click().run()
    assert not at.exception
    assert at.sidebar.number_input(key="f_max").value == 0.0 and "clear_limits" not in [b.key for b in at.button]


def test_first_visit_guidance_and_chat_example_buttons(tmp_path, monkeypatch):
    from src import data, pipeline, store
    from test_agent import agent_res

    res = agent_res(tmp_path, [{"name": "search", "args": {"request": "a quiet ice maker for my kitchen"}}],
                    [gen(["A", "B"])])
    monkeypatch.setattr(pipeline, "load_resources", lambda *a, **k: res)
    monkeypatch.setattr(store, "list_stores", lambda: [])
    monkeypatch.setattr(data, "load_interactions", lambda: pd.DataFrame(
        {"user_id": ["u_lg"] * 10, "product_id": ["B"] * 10, "rating": 5.0, "timestamp": range(10)}))
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
    assert any("How to get good results" in m.value for m in at.markdown)          # Shop: shown before a search
    at.button(key="chat-example-0").click().run()                                   # Chat: one click sends it
    assert not at.exception
    assert at.session_state["chat_log"][0] == {"role": "user", "text": "a quiet ice maker for my kitchen"}
    assert any(m.value.startswith("**1. ice maker a**") for m in at.markdown)
    assert "chat-example-0" not in [b.key for b in at.button]                       # examples only on a new chat
