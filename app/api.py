"""Read-only REST API: lets any store call the concierge, and serves the drop-in widget.

Run: .venv/Scripts/python.exe -m uvicorn app.api:app --port 8000
Store selection: ?store=<name> per request (stores built with `python -m src.store`); no store = benchmark catalog.
Binds to 127.0.0.1 by default. Before exposing it: HTTPS reverse proxy, the per-client rate limit
(app.rate_limit_per_minute), optionally API keys (app.require_api_key) and store webhooks (/webhooks/<store>/<platform>).
The request log and events keep no query text or IP; the shared LLM response cache does hold understood requests.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import queue
import threading
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from src import ROOT, load_config
from src import store as store_mod
from src.pipeline import (Resources, app_config, get_similar, load_resources, records, result_to_dict,
                          search_products)
from src.agent import ChatState, chat_turn, run_action
from src.store import STORES_DIR, link_template, valid_store_name

APP_DIR = Path(__file__).resolve().parent


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=500)                 # bounds LLM prompt size and cost
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


class EventsRequest(BaseModel):
    store: str | None = None
    client_id: str = Field(..., min_length=1, max_length=64)       # random per-browser id (widget localStorage)
    events: list[dict] = Field(default_factory=list, max_length=200)


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


MAX_WEBHOOK_BYTES = 2 * 1024 * 1024
MAX_BODY_BYTES = 256 * 1024                                 # every other request body
MAX_LOG_BYTES = 50 * 1024 * 1024                            # request log is rotated beyond this
CLIENT_IDS_PER_ADDRESS = 5                                  # browser ids accepted per address and day (events)
OPEN_PATHS = {"/health", "/widget.js", "/", "/docs", "/openapi.json", "/docs/oauth2-redirect", "/redoc"}
LLM_PATHS = {"/search", "/chat", "/search/stream", "/chat/stream"}
WEBHOOK_TOPICS = {"shopify": ({"products/create", "products/update"}, {"products/delete"}),
                  "woocommerce": ({"product.created", "product.updated", "product.restored"}, {"product.deleted"})}


class RateLimiter:
    """Sliding one-minute window per (client, group); returns the seconds to wait, 0 when allowed."""

    def __init__(self) -> None:
        self.hits: defaultdict[tuple, deque] = defaultdict(deque)
        self.lock = threading.Lock()
        self.last_sweep = 0.0

    def wait(self, key: tuple, limit: int, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        with self.lock:
            if now - self.last_sweep > 60:                # forget clients idle for a minute (bounded memory)
                for k in [k for k, v in self.hits.items() if not v or v[-1] <= now - 60]:
                    del self.hits[k]
                self.last_sweep = now
            q = self.hits[key]
            while q and q[0] <= now - 60:
                q.popleft()
            if len(q) >= limit:
                return 60 - (now - q[0])
            q.append(now)
            return 0.0


def ndjson_stream(work: Callable[[Callable[[dict], None]], dict]) -> StreamingResponse:
    """Phase 13.6: run `work(emit)` in a thread and stream each emitted event, then its final result, as one JSON
    object per line (products first, the validated answer when ready)."""
    events: queue.Queue = queue.Queue()

    def run() -> None:
        try:
            events.put(("final", work(lambda e: events.put(("event", e)))))
        except BaseException:  # noqa: BLE001 - never leave the stream waiting; details stay on the server
            events.put(("error", {"type": "error", "detail": "something went wrong; please try again"}))

    def lines():
        threading.Thread(target=run, daemon=True).start()
        while True:
            kind, item = events.get()
            yield json.dumps(item) + "\n"
            if kind != "event":
                return

    return StreamingResponse(lines(), media_type="application/x-ndjson")


def signature(secret: str, body: bytes) -> str:
    """Shopify and WooCommerce sign webhooks with base64(HMAC-SHA256(secret, raw body))."""
    return base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()


def create_app(loader: Callable[[str | None], Resources] | None = None,
               allowed_origins: list[str] | None = None, request_log: Path | None = None,
               rate_limits: dict[str, int] | None = None, require_api_key: bool | None = None,
               max_concurrent_llm: int | None = None) -> FastAPI:
    """Build the app; `loader(store)` returns Resources (cached per store and import). Tests pass a fake loader.

    LLM calls are capped per process (config app.max_llm_calls, NFR-3) and only listed origins may call the API
    from a browser (config app.allowed_origins or $CONCIERGE_ALLOWED_ORIGINS, comma-separated). Every client is
    rate-limited (app.rate_limit_per_minute); API keys are required when app.require_api_key is true; the real server
    logs each request (app.request_log) without query text or IP address.
    """
    cfg = load_config()
    app_cfg = cfg.get("app", {})
    if request_log is None and loader is None and app_cfg.get("request_log"):
        request_log = ROOT / app_cfg["request_log"]
    limits = {"llm": 30, "other": 120, "webhook": 600, "events": 300, **(app_cfg.get("rate_limit_per_minute") or {}),
              **(rate_limits or {})}
    shared: dict[str, object] = {}                         # one LLM client (call budget) per process
    need_key = app_cfg.get("require_api_key", False) if require_api_key is None else require_api_key
    trust_proxy = bool(app_cfg.get("trust_proxy", False))
    limiter, log_lock, cache_lock = RateLimiter(), threading.Lock(), threading.Lock()
    llm_slots = threading.BoundedSemaphore(int(max_concurrent_llm or app_cfg.get("max_concurrent_llm", 4)))
    ids_seen: dict[tuple[str, str], set] = {}              # (address, UTC day) -> browser ids; memory only
    seen_deliveries: OrderedDict[tuple[str, str], None] = OrderedDict()
    loader = loader or (lambda store: load_resources(app_config(cfg), max_calls=app_cfg.get("max_llm_calls"), store=store))
    env_origins = [o.strip() for o in os.environ.get("CONCIERGE_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    origins = allowed_origins if allowed_origins is not None else env_origins or app_cfg.get("allowed_origins", [])
    cache: dict[tuple[str | None, float], Resources] = {}

    def resources(store: str | None) -> Resources:
        store = store or None                                   # "" (e.g. ?store=) means the default catalog
        if store is not None and not valid_store_name(store):
            raise HTTPException(400, "invalid store name: use letters, digits, '-' or '_'")
        key = (store, store_version(store))
        with cache_lock:                                         # one load per store version, even when concurrent
            if key not in cache:
                try:
                    res = loader(store)
                except FileNotFoundError:
                    raise HTTPException(404, f"unknown store {store!r}; import it with python -m src.store") from None
                for old in [k for k in cache if k[0] == store]:   # an updated store replaces its old copy
                    del cache[old]
                if "client" in shared:
                    res.client = shared["client"]               # the per-process LLM cap survives reloads
                else:
                    shared["client"] = res.client
                cache[key] = res
            return cache[key]

    def client_address(request: Request) -> str:
        """The caller's address; behind a trusted proxy the last X-Forwarded-For entry (the one the proxy added)."""
        forwarded = request.headers.get("x-forwarded-for", "") if trust_proxy else ""
        return forwarded.split(",")[-1].strip() or (request.client.host if request.client else "?")

    def check_key(store: str | None, key: str | None) -> None:
        """When API keys are on: the store's key (stores/<name>/secrets.json) or CONCIERGE_API_KEY for the demo.
        Checked before the store is loaded; an unknown store and a wrong key look the same (401)."""
        if not need_key:
            return
        if store:
            expected = (store_mod.read_secrets(store) or {}).get("api_key")
        else:
            expected = os.environ.get("CONCIERGE_API_KEY")
        if not expected or not key or not hmac.compare_digest(key.encode(), expected.encode()):
            raise HTTPException(401, "missing or wrong API key (X-API-Key header)")

    app = FastAPI(title="Shopping Concierge API", version="1.0",
                  description="Semantic product search with enforced filters and grounded answers (read-only).")

    @app.middleware("http")
    async def limit_and_log(request: Request, call_next):
        """Per-client rate limit (429 + Retry-After) and one log line per request (no query text, no IP)."""
        start, path = time.perf_counter(), request.url.path
        group = None if path in OPEN_PATHS or request.method == "OPTIONS" else "webhook" \
            if path.startswith("/webhooks/") else "llm" if path in LLM_PATHS else \
            "events" if path == "/events" else "other"
        client = client_address(request)
        try:
            size = int(request.headers.get("content-length") or 0)
        except ValueError:
            return JSONResponse({"detail": "invalid Content-Length"}, status_code=400)
        cap = MAX_WEBHOOK_BYTES if group == "webhook" else MAX_BODY_BYTES
        wait = limiter.wait((client, group), int(limits[group])) if group else 0.0
        held = False
        if wait:
            response = JSONResponse({"detail": "too many requests; please try again shortly"}, status_code=429,
                                    headers={"Retry-After": str(int(wait) + 1)})
        elif size > cap:
            response = JSONResponse({"detail": "request body too large"}, status_code=413)
        elif group == "llm" and not llm_slots.acquire(blocking=False):
            response = JSONResponse({"detail": "the assistant is busy; please try again in a few seconds"},
                                    status_code=503, headers={"Retry-After": "5"})
        else:
            held = group == "llm"
            try:
                response = await call_next(request)
            except BaseException:
                if held:
                    llm_slots.release()
                raise
        if held and not path.endswith("/stream"):
            llm_slots.release()
            held = False
        if request_log is None and not held:
            return response

        def write(error: bool = False) -> None:
            line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "method": request.method, "path": path,
                    "store": request.query_params.get("store"), "status": response.status_code,
                    "ms": round((time.perf_counter() - start) * 1000, 1)}
            if error:
                line["stream_error"] = True
            if request_log is None:
                return
            with log_lock:
                log_path = Path(request_log)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if log_path.exists() and log_path.stat().st_size > MAX_LOG_BYTES:   # keep one older file
                    os.replace(log_path, log_path.with_name(log_path.name + ".1"))
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line) + "\n")

        if not path.endswith("/stream"):
            write()
            return response
        body = response.body_iterator                     # streams: log and free the LLM slot at the end

        async def logged():
            error = False
            try:
                async for chunk in body:
                    error = error or b'"type": "error"' in (chunk if isinstance(chunk, bytes) else chunk.encode())
                    yield chunk
            finally:
                if held:
                    llm_slots.release()
                write(error)

        response.body_iterator = logged()
        return response

    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],   # outermost, so
                       allow_headers=["Content-Type", "X-API-Key"])                            # 429s carry CORS

    @app.get("/health")
    async def health() -> dict:                     # on the event loop: answers even when worker threads are busy
        return {"status": "ok"}

    @app.get("/meta")
    def meta(store: str | None = None, x_api_key: str | None = Header(None)) -> dict:
        check_key(store or None, x_api_key)
        res = resources(store)
        return {"store": store or "benchmark", "products": len(res.catalog), "categories": res.meta.categories,
                "brands": res.meta.prompt_brands, "users": len(res.profiles)}

    @app.post("/search")
    def search(req: SearchRequest, x_api_key: str | None = Header(None)) -> dict:
        check_key(req.store or None, x_api_key)
        res = resources(req.store)
        if not req.query.strip():
            raise HTTPException(400, "Please enter a query.")               # spec 14: empty query
        out = search_products(req.query, res, max_price=req.max_price, min_price=req.min_price,
                              category=req.category, brand=req.brand, n=req.n, user_id=req.user_id)
        return result_to_dict(req.query, out, res)

    @app.post("/search/stream")
    def search_stream(req: SearchRequest, x_api_key: str | None = Header(None)) -> StreamingResponse:
        """Like /search, as NDJSON: {"type": "products", ...} first, then {"type": "answer", ...}."""
        check_key(req.store or None, x_api_key)
        res = resources(req.store)
        if not req.query.strip():
            raise HTTPException(400, "Please enter a query.")

        def work(emit):
            out = search_products(req.query, res, max_price=req.max_price, min_price=req.min_price,
                                  category=req.category, brand=req.brand, n=req.n, user_id=req.user_id,
                                  on_products=lambda view: emit({"type": "products", **view}))
            return {"type": "answer", **result_to_dict(req.query, out, res)}

        return ndjson_stream(work)

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest, x_api_key: str | None = Header(None)) -> StreamingResponse:
        """Like /chat, as NDJSON: for a search, {"type": "products", ...} first; always {"type": "turn", ...} last."""
        store = req.store or None
        check_key(store, x_api_key)
        res = resources(store)
        try:
            state = ChatState.model_validate(req.state or {})       # rejected with 400 before streaming starts
        except ValidationError:
            raise HTTPException(400, "invalid chat state") from None

        def work(emit):
            out = chat_turn(req.message, state, res, user_id=req.user_id, n=req.n, link=link_template(store),
                            on_products=lambda products: emit({"type": "products", "products": products}))
            return {"type": "turn", **out.to_dict()}

        return ndjson_stream(work)

    @app.post("/chat")
    def chat(req: ChatRequest, x_api_key: str | None = Header(None)) -> dict:
        """One chat turn: the LLM chooses an action, Python validates and runs it (src/agent.py)."""
        store = req.store or None
        check_key(store, x_api_key)
        res = resources(store)
        try:
            out = chat_turn(req.message, req.state, res, user_id=req.user_id, n=req.n, link=link_template(store))
        except ValidationError:
            raise HTTPException(400, "invalid chat state") from None
        return out.to_dict()

    @app.post("/chat/action")
    def chat_action(req: ChatAction, x_api_key: str | None = Header(None)) -> dict:
        """A card button (Save / Similar / Buy): runs directly, no LLM call."""
        store = req.store or None
        check_key(store, x_api_key)
        res = resources(store)
        try:
            out = run_action(req.action, req.args, req.state, res, n=req.n, link=link_template(store))
        except (ValueError, ValidationError) as exc:
            raise HTTPException(400, str(exc).splitlines()[0][:200]) from None
        return out.to_dict()

    @app.get("/similar/{product_id}")
    def similar(product_id: str, store: str | None = None, n: int = Query(5, ge=1, le=20),
                x_api_key: str | None = Header(None)) -> dict:
        check_key(store or None, x_api_key)
        res = resources(store)
        try:
            items = get_similar(product_id, res, n)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None
        return {"product_id": product_id, "similar": records(items)}

    @app.post("/webhooks/{store}/{platform}")
    async def webhook(store: str, platform: str, request: Request) -> dict:
        """Shopify / WooCommerce product webhooks: verify the signature, then add, update or remove that one
        product (only it is re-embedded). The daily --refresh remains the safety net for missed deliveries."""
        if platform not in WEBHOOK_TOPICS:
            raise HTTPException(404, "unknown platform; use shopify or woocommerce")
        if not valid_store_name(store):
            raise HTTPException(400, "invalid store name: use letters, digits, '-' or '_'")
        try:
            declared = int(request.headers.get("content-length") or 0)
        except ValueError:
            raise HTTPException(400, "invalid Content-Length") from None
        if declared > MAX_WEBHOOK_BYTES:
            raise HTTPException(413, "webhook body too large")
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_WEBHOOK_BYTES:
                raise HTTPException(413, "webhook body too large")
        secret = (store_mod.read_secrets(store) or {}).get("webhook_secret")
        h = request.headers
        sig = h.get("x-shopify-hmac-sha256" if platform == "shopify" else "x-wc-webhook-signature", "")
        if platform == "woocommerce" and not sig and body.startswith(b"webhook_id="):
            return {"status": "ping"}                     # WooCommerce's test delivery when a webhook is created
        if not secret or not sig or not hmac.compare_digest(sig.encode(), signature(secret, body).encode()):
            raise HTTPException(401, "invalid webhook signature")
        topic = h.get("x-shopify-topic" if platform == "shopify" else "x-wc-webhook-topic", "")
        delivery = h.get("x-shopify-webhook-id" if platform == "shopify" else "x-wc-webhook-delivery-id", "")
        if delivery and (store, delivery) in seen_deliveries:
            return {"status": "duplicate"}
        upserts, deletes = WEBHOOK_TOPICS[platform]
        if topic not in upserts | deletes:
            return {"status": "ignored", "topic": topic[:60]}
        try:
            payload = json.loads(body)
        except ValueError:
            raise HTTPException(400, "body is not JSON") from None
        if not isinstance(payload, dict):
            raise HTTPException(400, "body is not a JSON object")
        if platform == "woocommerce" and payload.get("type") == "variation":
            return {"status": "ignored", "topic": "variation"}   # the parent is updated by the daily refresh
        feed, delete_ids = None, []
        if topic in deletes:
            delete_ids = [str(payload.get("id"))]
        elif platform == "shopify":
            feed = store_mod.shopify_webhook_feed(payload)
        else:
            feed = store_mod.woocommerce_webhook_feed(payload)
        model = (await run_in_threadpool(resources, store)).model    # loaded off the event loop; reused
        try:
            log = await run_in_threadpool(store_mod.apply_store_changes, store, feed=feed, delete_ids=delete_ids,
                                          model=model,
                                          preset="shopify_json" if platform == "shopify" else "woocommerce")
        except TimeoutError:
            raise HTTPException(503, "store is being updated; retry", headers={"Retry-After": "30"}) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)[:200]) from None
        if delivery:
            seen_deliveries[(store, delivery)] = None
            while len(seen_deliveries) > 5000:
                seen_deliveries.popitem(last=False)
        log = log or {}
        if log.get("not_found"):
            return {"status": "not_found"}
        return {"status": "stale" if log.get("stale") and not log.get("applied") else "updated"}

    @app.post("/events")
    def events(req: EventsRequest, request: Request, x_api_key: str | None = Header(None)) -> dict:
        """Shopper events from the widget (shown / click / save / buy). Demo catalog events are not stored. One
        address may use at most CLIENT_IDS_PER_ADDRESS browser ids a day (kept in memory only, never stored), so
        rotating ids from one machine cannot inflate popularity."""
        store = req.store or None
        check_key(store, x_api_key)
        if store is None:
            return {"recorded": 0}
        if not valid_store_name(store):
            raise HTTPException(400, "invalid store name: use letters, digits, '-' or '_'")
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with cache_lock:
            for old in [k for k in ids_seen if k[1] != day]:
                del ids_seen[old]
            ids = ids_seen.setdefault((client_address(request), day), set())
            if req.client_id not in ids and len(ids) >= CLIENT_IDS_PER_ADDRESS:
                return {"recorded": 0}
            ids.add(req.client_id)
        try:
            return {"recorded": store_mod.record_events(store, req.client_id, req.events)}
        except ValueError:
            raise HTTPException(404, f"unknown store {store!r}") from None

    @app.get("/events/summary")
    def events_summary(store: str, days: int = Query(30, ge=1, le=365),
                       x_api_key: str | None = Header(None)) -> dict:
        """For the store owner only: the store's admin key (not the public widget key) is always required."""
        expected = (store_mod.read_secrets(store) or {}).get("admin_key")
        if not expected or not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
            raise HTTPException(401, "missing or wrong API key (X-API-Key header)")
        return store_mod.event_summary(store, days)

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
