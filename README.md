<div align="center">

# 🛒 AI Shopping Concierge

### Semantic Product Discovery, Personalized Recommendation & a Conversational Shopping Assistant

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%20%7C%20OpenAI%20%7C%20Claude-4285F4?logo=google&logoColor=white)](#-configuration-reference)
[![Embeddings](https://img.shields.io/badge/Embeddings-BGE--small-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![MCP](https://img.shields.io/badge/MCP-server-6E56CF)](#-mcp-server-for-ai-agents)
[![Tests](https://img.shields.io/badge/tests-304%20passing-2EA44F?logo=pytest&logoColor=white)](#-testing)

*Ask for "a quiet countertop ice maker under $150" and get products that really are under $150, ranked for you,
explained only with facts from the catalog.*

*atomcamp AI Bootcamp final project (NLP/LLM track) by Maaz Amjad, October 2026*

[Why](#-why-this-exists) · [Architecture](#-architecture) · [Results](#-results) · [Docs & notebooks](#-documentation--notebooks) ·
[Quick start](#-quick-start) · [Store integration](#-plug-into-a-real-store) · [API](#-api-reference) ·
[Project structure](#-project-structure) · [Testing](#-testing) · [Deployment](#-deployment)

</div>

---

## 💡 Why This Exists

Keyword search misses products described in other words, similarity search alone does not enforce limits such as
"under $80", and neither adapts to the shopper or explains its picks.

```
Keyword search:   "breathable jacket for rainy weather under $80"  ->  matches "jacket", ignores the budget
This concierge:   LLM extracts {max_price: 80}, Python enforces it, embeddings rank by meaning,
                  the answer quotes only catalog fields and is checked before it is shown
```

**Core rule: the LLM *extracts*, Python *enforces*.** The LLM never filters, ranks, or decides which products exist.

| Decision | Alternative | Why |
|---|---|---|
| LLM parses limits, Python applies them | Let the LLM pick products | Hard limits are guaranteed (100 % adherence on test), and invalid categories or brands are dropped, never guessed |
| Filter first, then semantic search inside the allowed products | Search first, filter after | Never runs short of results under tight limits; same quality, cheaper |
| Deterministic grounding validator on every answer | Trust the LLM | 0 invented product IDs; wrong prices or brands trigger one retry, then a template answer |
| Everything cached, seed 42, temperature 0 | Live calls | Every reported table regenerates without LLM calls |
| Production settings kept apart from the evaluated ones | One config | Improvements never silently change the reported benchmark |

---

## 🏗 Architecture

### System overview

```
                        ┌──────────────────────────── shopper ────────────────────────────┐
                        │  Streamlit app  ·  website chat bubble  ·  REST API  ·  MCP tools │
                        └───────────────────────────────┬──────────────────────────────────┘
                                                        │ request (+ optional shopper id)
┌───────────────────────────────────────────────────────▼───────────────────────────────────────────┐
│ 1 Query understanding   src/parse.py      LLM -> {semantic query, price, category, brand, "not X"} │
│ 2 Hard filters          src/filters.py    price · category · brand · exclusions · stock  (Python)  │
│ 3 Retrieval             src/retrieve.py   BGE embeddings, filter-first  (+ BM25 hybrid in the app) │
│ 4 Ranking               src/personalize.py semantic + personal + popularity (+ ratings in the app)  │
│ 5 Grounded answer       src/generate.py   structured LLM output -> validator -> text / template    │
│ ↺ Chat agent            src/agent.py      LLM picks one action (search, compare, save, buy...);    │
│                                           Python validates the arguments and runs it               │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
   src/llm.py: one client for every LLM call (Gemini, OpenAI-compatible, Claude): cache · rate limit ·
   retries · call cap · call log          src/store.py: imports, stock, webhooks, refresh, shopper events
```

### Request flow: one chat turn

```
Shopper           Widget / app        API (FastAPI)         Agent (Python)          LLM
  │  "cheaper"        │                    │                     │                    │
  │──────────────────>│  POST /chat/stream │                     │                    │
  │                   │───────────────────>│  chat_turn(state)   │                    │
  │                   │                    │────────────────────>│  forced tool call  │
  │                   │                    │                     │───────────────────>│
  │                   │                    │                     │<── search(max_price)│
  │                   │                    │                     │ merge earlier limits│
  │                   │                    │                     │ validate, filter,   │
  │                   │   products first   │<── products event ──│ retrieve, rank      │
  │  cards appear     │<───────────────────│                     │  generate answer   │
  │                   │                    │                     │───────────────────>│
  │                   │   validated answer │<── turn event ──────│<── grounded? ──────│
  │  summary appears  │<───────────────────│                     │                    │
```

### Fallback chain

```
parser output invalid -> retry once -> semantic-only search
answer fails validator -> regenerate once with the errors -> template answer from catalog fields
tool call unusable / LLM call cap reached -> plain search, answer from catalog fields
no product satisfies the limits -> say so and suggest which limit to relax (no LLM)
```

---

## 🧭 Design Decisions

**Filter-first instead of post-filter.** S4 (the proposal's post-filter) and S5 (filter-first) returned the same top
10 on every test query. **Decision:** keep filter-first; it cannot run out of candidates and is cheaper.

**Personalization measured honestly.** It predicts the next purchase far better than popularity (HitRate@10 6.3 % vs
2.8 %) but slightly lowers relevance for stated requests (NDCG@10 −0.026). **Decision:** keep it, report both.

**Products before the answer.** Generation dominates the wait (p95 12.7 s on the free tier) while retrieval takes
11 ms. **Decision:** stream the ranked products first and the answer once it has passed the validator; answers are
never streamed word by word, so unchecked text is never shown.

**Separate production settings.** Ratings, hybrid retrieval and the optional re-ranker live under `app:` in
`config.yaml`. **Decision:** the app gets the improvements; every benchmark table keeps the evaluated configuration.

**Report negative results.** A parser rule, an adaptive personal weight and a cross-encoder re-ranker did not hold up
on test. **Decision:** reverted or switched off, and documented in the [final report](docs/final_report.md).

---

## 📊 Results

15 held-out test queries, never used for tuning. Full analysis, ablations and error analysis in the
[final report](docs/final_report.md) and [04_evaluation](notebooks/04_evaluation.ipynb).

| System | P@10 | NDCG@10 [95% CI] | Constraint adherence@10 |
|---|---|---|---|
| S1 TF-IDF | 0.46 | 0.40 [0.24, 0.58] | 0.43 |
| S2 BM25 | 0.51 | 0.51 [0.33, 0.70] | 0.44 |
| S3 dense (BGE) | 0.67 | 0.65 [0.45, 0.82] | 0.63 |
| S4 dense + post-filter | 0.79 | 0.74 [0.57, 0.89] | 1.00 |
| S5 dense + filter-first | 0.79 | 0.74 [0.57, 0.89] | 1.00 |
| S5x + "not X" exclusions | 0.81 | 0.75 [0.59, 0.90] | 1.00 |
| S6 personalized | 0.79 | 0.71 [0.55, 0.87] | 1.00 |

| Goal | Target | Result |
|---|---|---|
| G1 query understanding | parse F1 ≥ 0.85 | ✅ **0.94** |
| G2 semantic retrieval | dense vs TF-IDF, with CIs | ✅ **+0.25** NDCG@10 [0.05, 0.48] |
| G3 constraint enforcement | adherence ≥ 95 % | ✅ **100 %** |
| G4 personalization | beat popularity on next item; no material relevance loss | ✅ 6.3 % vs 2.8 % · ⚠️ −0.026 NDCG@10 |
| G5 grounded answers | 0 IDs outside the retrieved set | ✅ **0**; 22/22 answers grounded |
| NFR-1 latency | retrieval p95 < 500 ms; end-to-end < 6 s | ✅ 11 ms · ❌ generation p95 12.7 s (free tier) |

### Production extensions (report section 12; each chosen on dev, reported once on test)

| Extension | Test result | In the app |
|---|---|---|
| ⭐ Star ratings in ranking | neutral (−0.009 [−0.032, +0.008]) | on |
| 📦 Stock awareness, incremental store refresh | sold-out products never recommended | on |
| 👤 Smaller personal weight for specific requests | dev and test disagreed | off |
| 🔐 Rate limits, API keys, signed store webhooks | replay and ordering guard, checked live | on |
| 🔤 Hybrid keyword + semantic retrieval | +0.037 [−0.017, +0.104]; part-number query 0.60 → 1.00 | on |
| ⚡ Products before the written answer | answers still validated first | on |
| 🏷 Parser rule for unstated categories | dropped real category requests | reverted |
| 📈 Shopper events feed popularity | privacy-limited; owner summary | on |
| 🔁 Cross-encoder re-ranker (bge-reranker-base) | −0.012 [−0.038, +0.010]; 13–17 s per query on CPU | off |

---

## 📚 Documentation & Notebooks

| Document | What it contains |
|---|---|
| [📄 Requirements](docs/requirements.md) | Goals, success criteria, functional and non-functional requirements, scope |
| [🛠 Technical specification](docs/tech_spec.md) | Architecture, data design, retrieval, ranking, grounding, evaluation design, interfaces |
| [📊 Final report](docs/final_report.md) | Results, ablations, personalization experiments, error analysis, limitations, production extensions |
| [📝 Proposal](docs/proposal.pdf) | The approved project proposal |
| [🎯 Intent](intent/intent.md) | Problem, outcome, users and constraints (planning stage) |

| Notebook | What it shows |
|---|---|
| [01 · EDA](notebooks/01_eda.ipynb) | The Amazon Appliances catalog: cleaning, price coverage, categories, brands, interactions |
| [02 · Baselines](notebooks/02_baselines.ipynb) | TF-IDF, BM25 and dense retrieval side by side |
| [03 · Pipeline](notebooks/03_pipeline.ipynb) | One request end to end: parsing, filters, retrieval, ranking, grounded answer |
| [04 · Evaluation](notebooks/04_evaluation.ipynb) | S1–S6 with confidence intervals, personalization experiments, error analysis |

Notebooks import from `src/` and run with an LLM call budget of zero (cached answers only):
`.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version / note | Needed for |
|---|---|---|
| Python | 3.12 (venv created with [uv](https://docs.astral.sh/uv/)) | everything |
| Gemini API key | free tier works (`GEMINI_API_KEY` in `.env`) | new LLM calls (cached ones need none) |
| Disk | ~2 GB (data, embeddings, models) | data and embeddings |
| Docker | optional | container deployment |

### Install and test

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Linux/macOS: .venv/bin/python
cp .env.example .env                                                   # then add your key after GEMINI_API_KEY=
.venv/Scripts/python.exe -m pytest -q                                  # 304 passed in a fresh clone, no API calls
```

### Get the data and build the index

```bash
B=https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main
curl -L -o data/raw/meta_Appliances.jsonl "$B/raw/meta_categories/meta_Appliances.jsonl"          # 285 MB
curl -L -o data/raw/Appliances_rating_only.csv "$B/benchmark/0core/rating_only/Appliances.csv"   # 122 MB
.venv/Scripts/python.exe -m src.data     # clean -> data/processed/ (+ eval/results/cleaning_log.json), ~20 s
.venv/Scripts/python.exe -m src.embed    # embeddings -> artifacts/ (~27 min on CPU, minutes on a GPU)
```

### Run it

```bash
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py      # demo app: Chat, Shop, Add your store
.venv/Scripts/python.exe -m uvicorn app.api:app --port 8000          # API + demo store page at http://localhost:8000/
```

The **Chat** tab is the shopping assistant: ask, refine ("cheaper", "only GE", "not stainless"), compare ("compare 1
and 2"), find similar items, keep a shortlist and open the product page to buy (no orders are placed). **Shop** is
single-shot search with the enforced limits, score components and the grounded answer. **Add your store** imports a
store, turns the website connection on and off, keeps products up to date and connects Claude Desktop, all with
buttons.

---

## 🏪 Plug Into a Real Store

### Import products

```bash
.venv/Scripts/python.exe -m src.store --name mystore --file products_export.csv --preset shopify
.venv/Scripts/python.exe -m src.store --name mystore --file https://mystore.com/products.json --preset shopify_json
.venv/Scripts/python.exe -m src.store --name mystore --file feed.csv --map title=Name,price=Cost     # any format
```

| Preset | Source | Stock read from |
|---|---|---|
| `shopify` | Admin "Export products" CSV | inventory quantity, "continue selling", untracked |
| `shopify_json` | `https://<store>/products.json` | each variant's availability |
| `woocommerce` | WooCommerce CSV export | "In stock?" and backorders |
| `generic` | `product_id,title,brand,category,price,description,features` | an `in_stock` / `availability` column |

Optional `rating` and `rating_count` columns are shown as stars and used in ranking. Out-of-stock products are never
recommended; unknown stock counts as available.

### Add the assistant to any page

```html
<div id="concierge-widget"></div>
<script src="https://YOUR-API/widget.js" data-store="mystore"></script>
```

A floating chat bubble; add `data-mode="search"` for a search box, `data-key="..."` when API keys are required, and
`data-track="off"` to disable shopper events. Add the store's domain to `app.allowed_origins` (or
`CONCIERGE_ALLOWED_ORIGINS`).

### Keep products current

| Method | How | Freshness |
|---|---|---|
| Scheduled refresh | `python -m src.store --name mystore --refresh [--every 1440]`, or *Update products now* in the app | the interval (daily recommended) |
| Webhooks | Shopify / WooCommerce product webhooks to `/webhooks/mystore/shopify` or `/woocommerce` | seconds |

Only products whose text changed are embedded again. Webhooks are checked with the platform's HMAC signature, repeats
are ignored, an older update never overwrites a newer one, and deleted products cannot come back from a late update.
WooCommerce: paste the store's secret from `--secrets` into the webhook form. Shopify signs with its own key: save it
with `--set-webhook-secret KEY` or in the app. Keep the daily refresh as a safety net for missed deliveries.

### Shopper events and popularity

The widget reports products shown, clicks, saves and Buy clicks (`POST /events`) with a random browser id renewed
every 30 days; no query text, no IP address, off under Do Not Track. An imported store's popularity becomes buy 5 +
save 3 + click 1 points, each browser counted once per product, action and day. The owner reads a 30-day summary
with `GET /events/summary?store=mystore` (owner `admin_key`) or `--events-summary`.

### 🤖 MCP server for AI agents

Read-only tools `search_products` and `get_similar`. The app's *Connect to Claude Desktop* button sets it up; by hand:

```json
{"mcpServers": {"shopping-assistant-mystore": {"command": "<project>/.venv/Scripts/python.exe",
  "args": ["<project>/app/mcp_server.py"], "env": {"CONCIERGE_STORE": "mystore"}}}}
```

---

## 🔌 API Reference

Interactive docs at `http://localhost:8000/docs`. `store` selects an imported store; omit it for the demo catalog.

| Method & path | Purpose | LLM calls |
|---|---|---|
| `POST /search` · `POST /search/stream` | One search: parsed limits, ranked products with score components, grounded answer (stream: products first) | 2 |
| `POST /chat` · `POST /chat/stream` | One chat turn; the client sends back the returned `state` | 1–2 |
| `POST /chat/action` | Card buttons (save, similar, buy) | 0 |
| `GET /similar/{product_id}` | Nearest products by embedding | 0 |
| `POST /events` · `GET /events/summary` | Shopper events · owner-only 30-day summary | 0 |
| `POST /webhooks/{store}/{shopify\|woocommerce}` | Signed product create / update / delete | 0 |
| `GET /meta` · `GET /health` · `GET /widget.js` · `GET /` | Store facts · health · widget · demo store page | 0 |

```bash
curl -s -X POST localhost:8000/search -H "Content-Type: application/json" \
     -d '{"query": "quiet countertop ice maker under $150", "n": 3}'
```

```json
{"query": "quiet countertop ice maker under $150",
 "parsed": {"semantic_query": "quiet countertop ice maker", "max_price": 150.0, "category": "Ice Makers", "...": "..."},
 "products": [{"product_id": "B0...", "title": "...", "price": 93.04, "avg_rating": 4.3, "s_sem": 1.0, "final": 0.93,
               "reason": "Operates quietly ...", "...": "..."}],
 "answer": "Here are quiet countertop ice makers under $150 ...", "flags": {"...": "..."}, "timings_ms": {"...": 0}}
```

```bash
curl -N -X POST localhost:8000/chat/stream -H "Content-Type: application/json" -d '{"message": "a beverage fridge, not stainless"}'
# {"type": "products", "products": [...]}        <- cards, as soon as they are ranked
# {"type": "turn", "text": "...", "state": {...}} <- validated answer and the state to send back next turn
```

---

## 📁 Project Structure

```
ai-shopping-concierge/
├── src/                 pipeline modules (logic lives here; notebooks import from it)
├── app/                 Streamlit app, FastAPI API, chat widget, MCP server, website connection helper
├── notebooks/           01_eda · 02_baselines · 03_pipeline · 04_evaluation
├── docs/                requirements · technical specification · final report · proposal
├── intent/              intent.md (planning stage)
├── eval/                benchmark queries, relevance labels, chat conversations, results/ (every reported number)
├── tests/               pytest suite (fake LLM client; never calls a real API)
├── config.yaml          all settings (seed 42, models, weights, app settings)
├── requirements.txt     pinned dependencies (CPU-only torch)
├── Dockerfile           API container (non-root, model baked in, offline at run time)
└── data/ · artifacts/ · stores/   downloaded data, embeddings and caches, imported stores (not committed)
```

| Module | Responsibility |
|---|---|
| `src/data.py` | Load and clean the catalog and interactions; product text |
| `src/embed.py` | BGE embeddings, saved and reloaded |
| `src/parse.py` | LLM query parser, catalog validation, fallback |
| `src/filters.py` | Hard filters (price, category, brand, exclusions, stock); relaxation suggestions |
| `src/retrieve.py` | TF-IDF, BM25, dense, post-filter, filter-first, hybrid |
| `src/personalize.py` | User profiles, weighted ranking, next-item experiment |
| `src/generate.py` | Grounded answers and the validator |
| `src/pipeline.py` | End-to-end `respond()`, `search_products()`, `get_similar()`, optional re-ranker |
| `src/agent.py` | Chat agent: tool choice, state, actions, conversation evaluation |
| `src/llm.py` | Shared LLM client and provider adapters |
| `src/store.py` | Store import, stock, refresh, webhooks, secrets, shopper events |
| `src/evaluate.py` | Metrics, bootstrap CIs, judge, qrels, experiments, report tables |
| `app/api.py` | REST API, streaming, rate limits, API keys, webhooks, events |
| `app/widget.js` | Drop-in chat bubble / search box (renders text only, never HTML) |
| `app/mcp_server.py` | MCP tools for AI agents |
| `app/streamlit_app.py` · `app/connection.py` | Demo app · starting the API and connecting Claude Desktop |

---

## ⚙️ Configuration Reference

### `config.yaml` (main keys)

| Key | Default | Meaning |
|---|---|---|
| `retrieval` | filter-first, pool 200 | Benchmark retrieval settings |
| `ranking` | `w_sem 0.7, w_pers 0.25, w_pop 0.05` | Evaluated ranking weights |
| `llm.provider` · `llm.model` · `llm.judge_model` | `gemini` · Flash-Lite · Flash | Provider (`gemini`, `openai` + optional `base_url`, `anthropic`) and models |
| `app.ranking` | adds `w_rat 0.1` | Production ranking (ratings on) |
| `app.retrieval` | `hybrid_alpha 0.5` | Production retrieval (hybrid on; optional `rerank`) |
| `app.max_llm_calls` | 300 | LLM calls per server process, then catalog-field answers |
| `app.allowed_origins` | localhost | Websites allowed to call the API from a browser |
| `app.rate_limit_per_minute` | 30 LLM / 120 other | Per-client limits (webhooks 600, events 300) |
| `app.require_api_key` · `app.trust_proxy` | false · false | API keys; trust `X-Forwarded-For` behind a proxy (then publish the port on 127.0.0.1 only) |
| `app.max_concurrent_llm` | 4 | LLM requests in progress at once; more get 503 + Retry-After |

### Environment variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` · `OPENAI_API_KEY` · `ANTHROPIC_API_KEY` | LLM keys (in `.env`, never committed) |
| `CONCIERGE_ALLOWED_ORIGINS` | Comma-separated allowed origins (overrides config) |
| `CONCIERGE_API_KEY` | API key for the demo catalog when keys are required |
| `CONCIERGE_STORE` | Store used by the MCP server |
| `HF_HUB_OFFLINE=1` | Use the cached embedding model without a network check |

### Dependencies

| Area | Packages |
|---|---|
| ML and retrieval | torch (CPU), sentence-transformers, faiss-cpu, scikit-learn, rank-bm25, scipy |
| Data | pandas, numpy, pydantic, pyyaml |
| LLM | google-genai, openai, anthropic, httpx |
| Apps | fastapi, uvicorn, streamlit, mcp |
| Notebooks and tests | jupyter, nbconvert, matplotlib, seaborn, pytest |

All versions are pinned in [requirements.txt](requirements.txt).

---

## 🧪 Testing

```bash
.venv/Scripts/python.exe -m pytest -q      # 304 passed
```

| Area | Test files |
|---|---|
| Data, embeddings, retrieval, filters | `test_data`, `test_embed`, `test_retrieve`, `test_filters` |
| Parsing, ranking, grounded answers | `test_parse`, `test_personalize`, `test_generate`, `test_pipeline` |
| Evaluation and judge | `test_evaluate`, `test_judge` |
| LLM client and providers | `test_llm`, `test_providers` |
| Chat agent | `test_agent` |
| Stores, webhooks, events | `test_store` |
| API, MCP, app | `test_api`, `test_mcp`, `test_connection`, `test_streamlit_app`, `test_smoke` |

**Mocking strategy:** every test uses a fake LLM client or fake SDK classes on a toy catalog; no test calls a real API.

### Reproduce the evaluation (no LLM calls: everything is cached)

```bash
.venv/Scripts/python.exe -m src.evaluate --split dev               # S1-S5 runs
.venv/Scripts/python.exe -m src.evaluate report --split test       # all test tables -> eval/results/test_*.csv
.venv/Scripts/python.exe -m src.personalize                        # next-item experiment
.venv/Scripts/python.exe -m src.pipeline                           # grounded answers -> eval/results/generation.csv
.venv/Scripts/python.exe -m src.agent --limit 0                    # scripted chat conversations
.venv/Scripts/python.exe -m src.evaluate hybrid --split test --completed   # production extensions (also rating, adaptive, rerank)
```

---

## 🛡 Error Handling & Security

| Layer | Situation | Behaviour |
|---|---|---|
| Parser | invalid JSON or unknown category / brand | one retry, then semantic-only; invalid values dropped and logged |
| Answer | invented ID, wrong price or brand | regenerate once, then a template answer from catalog fields |
| LLM | timeouts, 429, 503 | retries with backoff; per-process call cap; cached answers reused |
| Chat | unusable tool call, forged state | plain search fallback; client state re-validated every turn |
| API | abuse, cost | per-client rate limits; queries ≤ 500 characters, bodies ≤ 256 KB (webhooks 2 MB); at most `app.max_concurrent_llm` LLM requests at once, then 503 + Retry-After; `/health` never blocked; optional API keys; CORS allow-list |
| Store links | server-side request forgery | feed and Shopify addresses must resolve to public addresses, checked again on every redirect; 50 MB download cap |
| Webhooks | forged, replayed, out of order | HMAC check, duplicate ids ignored, older updates skipped, deletions remembered |
| Data | concurrent writers, crashes | lock file per store, atomic file swaps with rollback |
| Privacy | shopper data | request log and events keep no query text or IP; random browser id renewed every 30 days; events kept one year; at most 5 browser ids per address a day (checked in memory only). The LLM response cache does hold the understood requests (needed for reproducibility); clear `artifacts/llm_cache.jsonl` to remove them |
| Untrusted text | product descriptions, chat state | treated as data in prompts; widget renders text only |

---

## ☁️ Deployment

```bash
docker build -t concierge .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env \
  -v ${PWD}/data:/app/data -v ${PWD}/artifacts:/app/artifacts -v ${PWD}/stores:/app/stores concierge
```

The image runs as a non-root user, contains the embedding model and starts without network checks; data, embeddings,
caches and stores stay on mounted volumes. For a public deployment put an HTTPS reverse proxy in front, set
`app.trust_proxy: true` and add the store's domain to the allowed origins. Free-tier options: Hugging Face Spaces
(Docker) for a demo, or an always-free VM for persistent stores and webhooks. The Dockerfile has not yet been built
on the development machine.

---

## 🔧 Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| First request takes ~100 s | The embedding model checks Hugging Face on load | Set `HF_HUB_OFFLINE=1` once the model is cached |
| `429` / quota errors | Free-tier daily limits (failed calls count) | Wait for the daily reset; cached answers still work |
| Widget shows nothing on your site | Origin not allowed | Add the domain to `app.allowed_origins` |
| Webhooks return 401 | Wrong or missing signing key | Shopify: `--set-webhook-secret`; WooCommerce: use the secret from `--secrets` |
| A store import says "feed has no column" | Unrecognised export layout | Use `--map field=column,...` or the column questions in the app |
| `ImportError` after updating | The app was running old code | Restart the app |
| `503` "assistant is busy" | More LLM requests at once than `app.max_concurrent_llm` | Retry after a few seconds, or raise the limit |
| Streamlit reachable from other machines | Started outside the project folder, so `.streamlit/config.toml` was not read | Start it from the project folder, or add `--server.address localhost` |

---

## ✅ Production Checklist

- [x] Hard limits enforced in code; answers grounded and validated
- [x] Caching, retries, call caps, rate limits, optional API keys
- [x] Signed webhooks with replay and ordering guard; scheduled refresh as safety net
- [x] Stock awareness; privacy-limited shopper events
- [x] Container image definition (non-root, offline model)
- [ ] Build and run the Docker image on a server; HTTPS proxy
- [ ] Shared database and vector index for several server copies
- [ ] User accounts, per-store keys entered in the app, owner dashboard
- [ ] Online A/B test with a holdout group

---

## 📜 Data, Credits & License

Amazon Reviews 2023 (McAuley Lab): Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, 2024,
used for academic, non-commercial purposes. Embeddings: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
LLM: Google Gemini API. No license file is included: the code is © Maaz Amjad, shared for the atomcamp bootcamp
assessment; the dataset remains under its own terms.

<div align="center">

**[⬆ Back to top](#-ai-shopping-concierge)**

</div>
