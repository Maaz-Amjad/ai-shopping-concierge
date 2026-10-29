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
    assert cors.headers["access-control-allow-origin"] == "*"


def test_similar_and_widget_and_demo_page(client):
    sim = client.get("/similar/A", params={"n": 2}).json()
    assert [p["product_id"] for p in sim["similar"]] == ["B", "C"]
    assert client.get("/similar/zzz").status_code == 404
    js = client.get("/widget.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"] and "textContent" in js.text
    assert "innerHTML" not in js.text                                   # product text is never parsed as HTML
    assert 'src="/widget.js"' in client.get("/").text
