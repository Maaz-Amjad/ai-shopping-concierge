"""REST API over a toy catalog with a fake LLM (no model download, no real API calls)."""

import json

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from test_pipeline import gen, resources


@pytest.fixture
def client(tmp_path):
    parsed = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    res = resources(tmp_path, [parsed, gen(["A", "B"])])

    def loader(store):
        if store not in (None, "toy"):
            raise FileNotFoundError(store)
        return res

    return TestClient(create_app(loader))


def test_health_meta_and_unknown_store(client):
    assert client.get("/health").json() == {"status": "ok"}
    meta = client.get("/meta", params={"store": "toy"}).json()
    assert meta["products"] == 3 and meta["categories"] == ["Ice Makers", "Range Hoods"]
    assert client.get("/meta", params={"store": "nope"}).status_code == 404


def test_search_returns_filters_products_scores_and_answer(client):
    r = client.post("/search", json={"query": "ice maker", "n": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["parsed"]["category"] == "Ice Makers"
    assert [p["product_id"] for p in body["products"]] == ["A", "B"]
    assert body["products"][0]["reason"] == "fits" and "s_sem" in body["products"][0]
    assert body["answer"] and body["timings_ms"]["total"] >= 0
    assert r.headers.get("access-control-allow-origin") is None          # no Origin header sent
    cors = client.post("/search", json={"query": " "}, headers={"Origin": "https://shop.example"})
    assert cors.status_code == 400 and cors.json()["detail"] == "Please enter a query."
    assert "access-control-allow-origin" not in cors.headers                # origin not in config allow-list


def test_similar_and_widget_and_demo_page(client):
    sim = client.get("/similar/A", params={"n": 2}).json()
    assert [p["product_id"] for p in sim["similar"]] == ["B", "C"]
    assert client.get("/similar/zzz").status_code == 404
    js = client.get("/widget.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"] and "textContent" in js.text
    assert "innerHTML" not in js.text                                   # product text is never parsed as HTML
    assert 'src="/widget.js" data-store=""' in client.get("/").text
    page = client.get("/", params={"store": "toy"}).text
    assert 'data-store="toy"' in page and "Demo store: toy" in page
    assert client.get("/", params={"store": '"><script>'}).status_code == 400


# ---- verifier findings, Phase 11 (29 Oct) -------------------------------------------------------------

def test_path_like_store_names_are_rejected_before_loading(tmp_path):
    calls = []
    c = TestClient(create_app(lambda store: calls.append(store)))
    for bad in ("../x", "C:/Users/x", "a/b"):
        assert c.get("/meta", params={"store": bad}).status_code == 400
        assert c.post("/search", json={"query": "q", "store": bad}).status_code == 400
    assert calls == []                                                   # the loader never saw them


def test_cors_allows_only_configured_origins(tmp_path):
    c = TestClient(create_app(lambda store: resources(tmp_path, []), allowed_origins=["https://shop.example"]))
    ok = c.options("/search", headers={"Origin": "https://shop.example", "Access-Control-Request-Method": "POST"})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "https://shop.example"
    bad = c.options("/search", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_empty_store_parameter_means_the_default_catalog(client):
    # Verifier 11b: the app's preview link for the demo store is /?store= (empty) and returned 400.
    assert client.get("/", params={"store": ""}).status_code == 200
    assert client.get("/meta", params={"store": ""}).json()["store"] == "benchmark"


# ---- Phase 12: chat ------------------------------------------------------------------------------------

def chat_client(tmp_path, tool_calls, answers):
    from test_agent import agent_res

    res = agent_res(tmp_path, tool_calls, answers)
    return TestClient(create_app(lambda store: res))


def test_chat_turns_carry_state_and_buttons_need_no_llm(tmp_path):
    c = chat_client(tmp_path, [{"name": "search", "args": {"request": "ice maker", "category": "Ice Makers"}}],
                    [gen(["A", "B"])])
    first = c.post("/chat", json={"message": "an ice maker please"}).json()
    assert first["action"] == "search" and [p["number"] for p in first["products"]] == [1, 2]
    assert first["products"][0]["url"] == "https://www.amazon.com/dp/A"            # benchmark hand-off link
    saved = c.post("/chat/action", json={"action": "save", "args": {"items": [2]}, "state": first["state"]}).json()
    assert saved["state"]["shortlist"] == ["B"]
    assert c.post("/chat/action", json={"action": "checkout", "state": saved["state"]}).status_code == 400
    assert c.post("/chat", json={"message": "hi", "state": {"shown": "not a list"}}).status_code == 400
    assert c.post("/chat", json={"message": "hi", "store": "../x"}).status_code == 400


def test_widget_card_buttons_send_the_product_id(client):
    js = client.get("/widget.js").text
    assert "product_id: p.product_id" in js and "{item: p.number}" not in js


def test_widget_shows_star_ratings(client):
    js = client.get("/widget.js").text
    assert "function stars(p)" in js and js.count("stars(p)") >= 3



# ---- Phase 13.4: webhooks, rate limit, API key, request log ---------------------------------------------

import base64  # noqa: E402
import hashlib  # noqa: E402
import hmac  # noqa: E402


def signed(secret: str, payload: dict) -> tuple[bytes, str]:
    body = json.dumps(payload).encode()
    return body, base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()


@pytest.fixture
def shop_app(tmp_path, monkeypatch):
    from src import store
    from test_store import SHOP_PRODUCT, FakeModel, webhook_store

    webhook_store(tmp_path, monkeypatch)
    res = resources(tmp_path, [])
    res.model = FakeModel()
    calls = []
    monkeypatch.setattr(store, "apply_store_changes",
                        lambda name, feed=None, delete_ids=(), model=None, cfg=None, preset=None: calls.append(
                            (name, None if feed is None else feed.to_dict("records"), list(delete_ids))) or {})
    app = create_app(lambda s: res, request_log=tmp_path / "requests.jsonl")
    return TestClient(app), store.store_secrets("shop")["webhook_secret"], calls, SHOP_PRODUCT, tmp_path


def test_signed_shopify_webhook_updates_the_store(shop_app):
    client, secret, calls, product, _ = shop_app
    body, sig = signed(secret, product)
    headers = {"X-Shopify-Hmac-Sha256": sig, "X-Shopify-Topic": "products/update", "X-Shopify-Webhook-Id": "w1",
               "Content-Type": "application/json"}
    assert client.post("/webhooks/shop/shopify", content=body, headers=headers).status_code == 200
    assert calls[0][0] == "shop" and calls[0][1][0]["handle"] == "steel-kettle"
    assert client.post("/webhooks/shop/shopify", content=body, headers=headers).json()["status"] == "duplicate"
    body, sig = signed(secret, {"id": 102})
    client.post("/webhooks/shop/shopify", content=body, headers={**headers, "X-Shopify-Hmac-Sha256": sig,
                                                                   "X-Shopify-Topic": "products/delete",
                                                                   "X-Shopify-Webhook-Id": "w2"})
    assert calls[1] == ("shop", None, ["102"]) and len(calls) == 2


def test_webhooks_reject_bad_signatures_unknown_stores_and_topics(shop_app):
    client, secret, calls, product, _ = shop_app
    body, sig = signed(secret, product)
    bad = client.post("/webhooks/shop/shopify", content=body,
                      headers={"X-Shopify-Hmac-Sha256": ("A" if sig[0] != "A" else "B") + sig[1:],
                               "X-Shopify-Topic": "products/update"})              # always differs
    assert bad.status_code == 401
    assert client.post("/webhooks/nope/shopify", content=body,          # 401, not 404: store names stay private
                       headers={"X-Shopify-Hmac-Sha256": sig, "X-Shopify-Topic": "products/update"}).status_code == 401
    assert client.post("/webhooks/shop/magento", content=body, headers={}).status_code == 404
    other = client.post("/webhooks/shop/shopify", content=body,
                        headers={"X-Shopify-Hmac-Sha256": sig, "X-Shopify-Topic": "orders/create"})
    assert other.status_code == 200 and other.json()["status"] == "ignored" and calls == []


def test_woocommerce_webhook_signature_and_ping(shop_app):
    client, secret, calls, _, _ = shop_app
    ping = client.post("/webhooks/shop/woocommerce", content=b"webhook_id=12",
                       headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert ping.status_code == 200 and calls == []
    body, sig = signed(secret, {"id": 7, "sku": "K-7", "name": "Kettle", "status": "publish",
                                "stock_status": "instock", "regular_price": "20"})
    ok = client.post("/webhooks/shop/woocommerce", content=body,
                     headers={"X-WC-Webhook-Signature": sig, "X-WC-Webhook-Topic": "product.updated",
                              "X-WC-Webhook-Delivery-ID": "d1"})
    assert ok.status_code == 200 and calls[0][1][0]["SKU"] == "K-7"


def test_rate_limit_api_key_and_request_log(tmp_path, monkeypatch):
    parsed = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    res = resources(tmp_path, [parsed, gen(["A", "B"])] * 5)
    monkeypatch.setenv("CONCIERGE_API_KEY", "k" * 32)
    app = create_app(lambda s: res, request_log=tmp_path / "requests.jsonl", rate_limits={"llm": 2, "other": 3},
                     require_api_key=True)
    client = TestClient(app)
    assert client.get("/meta").status_code == 401                                  # key required
    key = {"X-API-Key": "k" * 32}
    assert client.get("/meta", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/health").status_code == 200                                # always open
    assert [client.post("/search", json={"query": "secret words"}, headers=key).status_code
            for _ in range(3)] == [200, 200, 429]
    assert "Retry-After" in client.post("/search", json={"query": "x"}, headers=key).headers
    lines = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert {"ts", "method", "path", "status", "ms"} <= set(lines[0])
    assert "secret words" not in (tmp_path / "requests.jsonl").read_text()          # no query text logged
    assert [x["status"] for x in lines if x["path"] == "/search"][:3] == [200, 200, 429]


def test_widget_sends_the_store_api_key_when_given(client):
    js = client.get("/widget.js").text
    assert 'getAttribute("data-key")' in js and 'headers["X-API-Key"]' in js and js.count("headers: headers") == 2



# ---- verifier findings, Phase 13.4 ----------------------------------------------------------------------

def test_rate_limiter_forgets_idle_clients():
    from app.api import RateLimiter

    lim = RateLimiter()
    for i in range(50):
        lim.wait((f"ip{i}", "llm"), 5, now=0.0)
    lim.wait(("late", "llm"), 5, now=200.0)
    assert set(lim.hits) == {("late", "llm")}


def test_forwarded_for_uses_the_address_added_by_the_trusted_proxy(tmp_path, monkeypatch):
    from app import api

    monkeypatch.setattr(api, "load_config", lambda: {"app": {"trust_proxy": True}})
    res = resources(tmp_path, [])
    client = TestClient(create_app(lambda s: res, rate_limits={"other": 2}))
    codes = [client.get("/meta", headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.9"}).status_code
             for i in range(3)]
    assert codes == [200, 200, 429]                                    # spoofed first entries do not help


def test_api_key_is_checked_before_loading_and_hides_which_stores_exist(tmp_path, monkeypatch):
    from src import store

    monkeypatch.setattr(store, "STORES_DIR", tmp_path / "stores")
    loads = []
    client = TestClient(create_app(lambda s: loads.append(s), require_api_key=True))
    assert client.get("/meta", params={"store": "nope"}).status_code == 401
    assert loads == [] and not (tmp_path / "stores").exists()


def test_store_api_key_path(tmp_path, monkeypatch):
    from src import store
    from test_store import webhook_store

    webhook_store(tmp_path, monkeypatch)
    key = store.store_secrets("shop")["api_key"]
    res = resources(tmp_path, [])
    client = TestClient(create_app(lambda s: res, require_api_key=True))
    assert client.get("/meta", params={"store": "shop"}).status_code == 401
    assert client.get("/meta", params={"store": "shop"}, headers={"X-API-Key": key}).status_code == 200


def test_webhook_limits_body_size_and_hides_unknown_stores(shop_app):
    client, secret, calls, product, _ = shop_app
    big = b"{" + b" " * (3 * 1024 * 1024) + b"}"
    assert client.post("/webhooks/shop/shopify", content=big,
                       headers={"X-Shopify-Hmac-Sha256": "x", "X-Shopify-Topic": "products/update"}).status_code == 413
    body, sig = signed(secret, product)
    assert client.post("/webhooks/nope/shopify", content=body,
                       headers={"X-Shopify-Hmac-Sha256": sig, "X-Shopify-Topic": "products/update"}).status_code == 401


def test_woocommerce_variation_deliveries_are_ignored(shop_app):
    client, secret, calls, _, _ = shop_app
    body, sig = signed(secret, {"id": 9, "type": "variation", "parent_id": 7, "status": "publish"})
    r = client.post("/webhooks/shop/woocommerce", content=body,
                    headers={"X-WC-Webhook-Signature": sig, "X-WC-Webhook-Topic": "product.updated"})
    assert r.json()["status"] == "ignored" and calls == []


def test_reloaded_store_replaces_the_old_copy_and_keeps_the_llm_budget(tmp_path, monkeypatch):
    from app import api

    versions = iter([1.0, 2.0])                                         # the store changed between calls
    monkeypatch.setattr(api, "store_version", lambda store: next(versions))
    made = []

    def loader(store):
        res = resources(tmp_path / str(len(made)), [])
        made.append(res)
        return res

    client = TestClient(create_app(loader))
    client.get("/meta")
    client.get("/meta")
    assert len(made) == 2 and made[1].client is made[0].client          # one LLM budget per process


def test_stale_webhook_is_reported(shop_app, monkeypatch):
    from src import store

    client, secret, calls, product, _ = shop_app
    monkeypatch.setattr(store, "apply_store_changes", lambda *a, **k: {"applied": 0, "stale": ["steel-kettle"]})
    body, sig = signed(secret, product)
    r = client.post("/webhooks/shop/shopify", content=body,
                    headers={"X-Shopify-Hmac-Sha256": sig, "X-Shopify-Topic": "products/update"})
    assert r.status_code == 200 and r.json()["status"] == "stale"


def test_search_and_chat_stream_products_first_then_the_answer(tmp_path):
    from test_agent import agent_res

    parsed = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    res = agent_res(tmp_path, [{"name": "search", "args": {"request": "ice maker", "category": "Ice Makers"}}],
                    [parsed, gen(["A", "B"]), gen(["A", "B"])])
    client = TestClient(create_app(lambda s: res))
    lines = [json.loads(x) for x in client.post("/search/stream", json={"query": "ice maker"}).text.splitlines()]
    assert [e["type"] for e in lines] == ["products", "answer"]
    assert [p["product_id"] for p in lines[0]["products"]] == ["A", "B"] and lines[1]["answer"]
    chat = [json.loads(x) for x in client.post("/chat/stream", json={"message": "ice maker"}).text.splitlines()]
    assert [e["type"] for e in chat] == ["products", "turn"] and chat[1]["state"]["shown"] == ["A", "B"]
    assert chat[0]["products"][0]["number"] == 1


def test_widget_reads_the_streams(client):
    js = client.get("/widget.js").text
    assert "/chat/stream" in js and "/search/stream" in js and "getReader" in js


# ---- verifier findings, Phase 13.6 ----------------------------------------------------------------------

def test_stream_sends_products_once_when_the_call_cap_hits_during_generation(tmp_path):
    parsed = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    res = resources(tmp_path, [parsed, gen(["A", "B"])])
    res.client.max_calls = 1                                          # parse allowed, generation refused
    client = TestClient(create_app(lambda s: res))
    lines = [json.loads(x) for x in client.post("/search/stream", json={"query": "ice maker"}).text.splitlines()]
    assert [e["type"] for e in lines] == ["products", "answer"]
    assert [p["product_id"] for p in lines[1]["products"]] == ["A", "B"]          # category limit kept
    assert lines[1]["parsed"]["category"] == "Ice Makers"


def test_stream_no_result_has_no_products_event_and_errors_are_generic(tmp_path, monkeypatch):
    from app import api

    parsed = json.dumps({"semantic_query": "toaster", "category": "Ice Makers", "max_price": 1})
    res = resources(tmp_path, [parsed])
    client = TestClient(create_app(lambda s: res))
    lines = [json.loads(x) for x in client.post("/search/stream", json={"query": "toaster"}).text.splitlines()]
    assert [e["type"] for e in lines] == ["answer"]
    monkeypatch.setattr(api, "search_products", lambda *a, **k: (_ for _ in ()).throw(KeyError("C:/secret/path")))
    err = [json.loads(x) for x in client.post("/search/stream", json={"query": "x"}).text.splitlines()]
    assert err == [{"type": "error", "detail": "something went wrong; please try again"}]
    assert client.post("/chat/stream", json={"message": "hi", "state": {"shown": "not a list"}}).status_code == 400


def test_stream_routes_are_logged_when_the_stream_ends(tmp_path, monkeypatch):
    import time as _time

    from app import api

    parsed = json.dumps({"semantic_query": "ice maker", "category": "Ice Makers"})
    res = resources(tmp_path, [parsed, gen(["A", "B"])])
    real = api.search_products

    def slow(*a, **k):
        _time.sleep(0.3)
        return real(*a, **k)

    monkeypatch.setattr(api, "search_products", slow)
    client = TestClient(create_app(lambda s: res, request_log=tmp_path / "log.jsonl"))
    client.post("/search/stream", json={"query": "ice maker"})
    line = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert line["path"] == "/search/stream" and line["ms"] >= 300


def test_widget_stream_cleanup_final_event_and_fallback(client):
    js = client.get("/widget.js").text
    assert "early.remove()" in js.split(".catch(")[1]                  # early cards removed on errors too
    assert 'last.type !== "answer" && last.type !== "turn"' in js     # a cut stream is an error
    assert "ReadableStream" in js                                     # plain endpoint when streams are missing
