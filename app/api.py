"""Read-only REST API: lets any store call the concierge, and serves the drop-in widget.

Run: .venv/Scripts/python.exe -m uvicorn app.api:app --port 8000
Store selection: ?store=<name> per request (stores built with `python -m src.store`); no store = benchmark catalog.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.pipeline import Resources, get_similar, load_resources, records, result_to_dict, search_products

APP_DIR = Path(__file__).resolve().parent


class SearchRequest(BaseModel):
    query: str
    store: str | None = None
    user_id: str | None = None
    n: int = Field(5, ge=1, le=20)
    max_price: float | None = None
    min_price: float | None = None
    category: str | None = None
    brand: str | None = None


def create_app(loader: Callable[[str | None], Resources] | None = None) -> FastAPI:
    """Build the app; `loader(store)` returns Resources (cached per store). Tests pass a fake loader."""
    loader = loader or (lambda store: load_resources(store=store))
    cache: dict[str | None, Resources] = {}

    def resources(store: str | None) -> Resources:
        if store not in cache:
            try:
                cache[store] = loader(store)
            except FileNotFoundError:
                raise HTTPException(404, f"unknown store {store!r}; import it with python -m src.store") from None
        return cache[store]

    app = FastAPI(title="Shopping Concierge API", version="1.0",
                  description="Semantic product search with enforced filters and grounded answers (read-only).")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/meta")
    def meta(store: str | None = None) -> dict:
        res = resources(store)
        return {"store": store or "benchmark", "products": len(res.catalog), "categories": res.meta.categories,
                "brands": res.meta.prompt_brands, "users": len(res.profiles)}

    @app.post("/search")
    def search(req: SearchRequest) -> dict:
        if not req.query.strip():
            raise HTTPException(400, "Please enter a query.")               # spec 14: empty query
        res = resources(req.store)
        out = search_products(req.query, res, max_price=req.max_price, min_price=req.min_price,
                              category=req.category, brand=req.brand, n=req.n, user_id=req.user_id)
        return result_to_dict(req.query, out, res)

    @app.get("/similar/{product_id}")
    def similar(product_id: str, store: str | None = None, n: int = Query(5, ge=1, le=20)) -> dict:
        try:
            items = get_similar(product_id, resources(store), n)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None
        return {"product_id": product_id, "similar": records(items)}

    @app.get("/widget.js")
    def widget() -> FileResponse:
        return FileResponse(APP_DIR / "widget.js", media_type="application/javascript")

    @app.get("/", response_class=HTMLResponse)
    def demo_page() -> str:
        """A blank 'store page' with the widget pasted in, exactly as a store owner would."""
        return (APP_DIR / "demo_store.html").read_text(encoding="utf-8")

    return app


app = create_app()
