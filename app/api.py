"""Read-only REST API: lets any store call the concierge, and serves the drop-in widget.

Run: .venv/Scripts/python.exe -m uvicorn app.api:app --port 8000
Store selection: ?store=<name> per request (stores built with `python -m src.store`); no store = benchmark catalog.
Binds to 127.0.0.1 by default; put it behind a reverse proxy with authentication before exposing it publicly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from src import load_config
from src.pipeline import Resources, get_similar, load_resources, records, result_to_dict, search_products
from src.agent import chat_turn, run_action
from src.store import STORES_DIR, link_template, valid_store_name

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


class ChatRequest(BaseModel):
    message: str = Field("", max_length=2000)
    store: str | None = None
    user_id: str | None = None
    n: int = Field(5, ge=1, le=10)
    state: dict | None = None          # returned by the previous turn; validated again on the server


class ChatAction(BaseModel):
    action: str                        # save | similar | buy | remove | show_shortlist (buttons; no LLM call)
    args: dict = Field(default_factory=dict)
    store: str | None = None
    n: int = Field(5, ge=1, le=10)
    state: dict | None = None


def store_version(store: str | None) -> float:
    """Changes when a store is re-imported, so the API reloads it."""
    path = STORES_DIR / store / "catalog.pkl" if store else None
    return path.stat().st_mtime if path is not None and path.exists() else 0.0


def create_app(loader: Callable[[str | None], Resources] | None = None,
               allowed_origins: list[str] | None = None) -> FastAPI:
    """Build the app; `loader(store)` returns Resources (cached per store and import). Tests pass a fake loader.

    LLM calls are capped per process (config app.max_llm_calls, NFR-3) and only listed origins may call the API
    from a browser (config app.allowed_origins or $CONCIERGE_ALLOWED_ORIGINS, comma-separated).
    """
    cfg = load_config()
    app_cfg = cfg.get("app", {})
    loader = loader or (lambda store: load_resources(cfg, max_calls=app_cfg.get("max_llm_calls"), store=store))
    env_origins = [o.strip() for o in os.environ.get("CONCIERGE_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    origins = allowed_origins if allowed_origins is not None else env_origins or app_cfg.get("allowed_origins", [])
    cache: dict[tuple[str | None, float], Resources] = {}

    def resources(store: str | None) -> Resources:
        store = store or None                                   # "" (e.g. ?store=) means the default catalog
        if store is not None and not valid_store_name(store):
            raise HTTPException(400, "invalid store name: use letters, digits, '-' or '_'")
        key = (store, store_version(store))
        if key not in cache:
            try:
                cache[key] = loader(store)
            except FileNotFoundError:
                raise HTTPException(404, f"unknown store {store!r}; import it with python -m src.store") from None
        return cache[key]

    app = FastAPI(title="Shopping Concierge API", version="1.0",
                  description="Semantic product search with enforced filters and grounded answers (read-only).")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

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

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        """One chat turn: the LLM chooses an action, Python validates and runs it (src/agent.py)."""
        store = req.store or None
        res = resources(store)
        try:
            out = chat_turn(req.message, req.state, res, user_id=req.user_id, n=req.n, link=link_template(store))
        except ValidationError:
            raise HTTPException(400, "invalid chat state") from None
        return out.to_dict()

    @app.post("/chat/action")
    def chat_action(req: ChatAction) -> dict:
        """A card button (Save / Similar / Buy): runs directly, no LLM call."""
        store = req.store or None
        res = resources(store)
        try:
            out = run_action(req.action, req.args, req.state, res, n=req.n, link=link_template(store))
        except (ValueError, ValidationError) as exc:
            raise HTTPException(400, str(exc).splitlines()[0][:200]) from None
        return out.to_dict()

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
    def demo_page(store: str | None = None) -> str:
        """A blank 'store page' with the widget pasted in, exactly as a store owner would (?store=<name>)."""
        store = store or None
        if store is not None and not valid_store_name(store):
            raise HTTPException(400, "invalid store name: use letters, digits, '-' or '_'")
        page = (APP_DIR / "demo_store.html").read_text(encoding="utf-8")
        return page.replace("{{STORE}}", store or "").replace("{{STORE_LABEL}}", f"Demo store: {store or 'appliances'}")

    return app


app = create_app()
