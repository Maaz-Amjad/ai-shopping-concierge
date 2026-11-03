<div align="center">

# 🛒 AI Shopping Concierge

### Semantic Product Discovery, Personalized Recommendation & a Conversational Shopping Assistant

[![Live demo](https://img.shields.io/badge/Live%20demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://ai-shopping-concierge.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%20%7C%20OpenAI%20%7C%20Claude-4285F4?logo=google&logoColor=white)](#-built-with)
[![Embeddings](https://img.shields.io/badge/Embeddings-BGE--small-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![MCP](https://img.shields.io/badge/MCP-server-6E56CF)](#-store-integration)
[![Docker](https://img.shields.io/badge/Docker-ai--shopping--concierge-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/maazamjad110/ai-shopping-concierge)
[![Tests](https://img.shields.io/badge/tests-323%20passing-2EA44F?logo=pytest&logoColor=white)](#-testing)

*Ask for "a quiet countertop ice maker under $150" and get products that really are under $150, ranked for you,
explained only with facts from the catalog.*

### [▶ Try the live demo](https://ai-shopping-concierge.streamlit.app/)

*atomcamp AI Bootcamp final project (NLP/LLM track) by Maaz Amjad, 2026*

[Why](#-why-this-exists) · [Phases](#-project-phases) · [Docs](#-documentation--notebooks) · [Structure](#-project-structure) · [Architecture](#-architecture) ·
[Results](#-results) · [Live demo](#-live-demo) · [Deployment](#-deployment) · [Store integration](#-store-integration) · [Security](#-error-handling--security)

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
| LLM parses limits, Python applies them | Let the LLM pick products | Hard limits are guaranteed (100 % adherence on test); invalid categories or brands are dropped, never guessed |
| Filter first, then semantic search inside the allowed products | Search first, filter after | Never runs short of results under tight limits; same quality, cheaper |
| Deterministic grounding validator on every answer | Trust the LLM | 0 invented product IDs; wrong prices or brands trigger one retry, then a template answer |
| Everything cached, seed 42, temperature 0 | Live calls | Every reported table regenerates without LLM calls |
| Production settings kept apart from the evaluated ones | One config | Improvements never silently change the reported benchmark |

---

## 🗂 Project Phases

| Phase | What was built |
|---|---|
| 0 · Plan | Requirements, technical specification, plan and success criteria |
| 1 · Setup | Repository, pinned environment, shared LLM client (cache, rate limit, retries, call cap), safety hooks |
| 2 · Data | Amazon Appliances catalog: 94k listings cleaned to 10,000 products; exploratory analysis |
| 3 · Embeddings & baselines | BGE embeddings; TF-IDF, BM25 and dense retrieval compared |
| 4 · Query understanding | LLM parser for price, category, brand and "not X"; hard filters in Python |
| 5 · Evaluation harness | Metrics, bootstrap confidence intervals, 25-query benchmark split into dev and test |
| 6 · Relevance labels | 659 query–product pairs graded by an LLM judge; 25 % checked by hand (92 % agreement) |
| 7 · Personalization | Shopper profiles from purchase history; next-purchase and relevance experiments |
| 8 · Grounded answers | Structured LLM answers checked for invented products, prices and brands |
| 9 · Test evaluation | S1–S6 on the held-out test set, ablations and error analysis |
| 10 · Report | Final report, four notebooks, documentation |
| 11 · Apps & integrations | No-code Streamlit app, REST API, website widget, store import (Shopify, WooCommerce, CSV/JSON), MCP server |
| 12 · Chat assistant | Multi-turn shopping chat (refine, compare, shortlist, buy link); Gemini, OpenAI or Claude |
| 13 · Production features | Nine extensions, each chosen on dev and reported once on test (see [Results](#-results)) |
| 14 · Deployment | Public demo on Streamlit Community Cloud where visitors bring their own AI key; container image on Docker Hub |

---

## 📚 Documentation & Notebooks

| Document | What it contains |
|---|---|
| [📄 Requirements](docs/requirements.md) | Goals, success criteria, functional and non-functional requirements, scope |
| [🛠 Technical specification](docs/tech_spec.md) | Architecture, data design, retrieval, ranking, grounding, evaluation design, interfaces |
| [📊 Final report](docs/final_report.md) | Results, ablations, personalization experiments, error analysis, limitations, production extensions |
| [📝 Proposal](docs/proposal.pdf) | The project proposal |
| [🎯 Intent](intent/intent.md) | Problem, outcome, users and constraints |

| Notebook | What it shows |
|---|---|
| [01 · EDA](notebooks/01_eda.ipynb) | The Amazon Appliances catalog: cleaning, price coverage, categories, brands, interactions |
| [02 · Baselines](notebooks/02_baselines.ipynb) | TF-IDF, BM25 and dense retrieval side by side |
| [03 · Pipeline](notebooks/03_pipeline.ipynb) | One request end to end: parsing, filters, retrieval, ranking, grounded answer |
| [04 · Evaluation](notebooks/04_evaluation.ipynb) | S1–S6 with confidence intervals, personalization experiments, error analysis |

---

## 📜 Data, Credits & License

Amazon Reviews 2023 (McAuley Lab): Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, 2024,
used for academic, non-commercial purposes; the processed catalog, anonymized interactions and embeddings in this
repository remain under the dataset's terms. Embeddings: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
The code is © Maaz Amjad, shared for the atomcamp bootcamp assessment.

---

## 📁 Project Structure

```
ai-shopping-concierge/
├── src/                 pipeline modules (notebooks import from here)
├── app/                 Streamlit app, FastAPI API, chat widget, MCP server
├── notebooks/           01_eda · 02_baselines · 03_pipeline · 04_evaluation
├── docs/                requirements · technical specification · final report · proposal
├── intent/              intent.md
├── eval/                benchmark queries, relevance labels, chat conversations, results (every reported number)
├── tests/               pytest suite (fake LLM client; never calls a real API)
├── data/ · artifacts/   processed catalog and embeddings
├── config.yaml          all settings (seed 42, models, weights, app settings)
└── Dockerfile           API container (non-root, model included)
```

| Module | Responsibility |
|---|---|
| `src/data.py` · `src/embed.py` | Clean the catalog and interactions; BGE embeddings |
| `src/parse.py` · `src/filters.py` | LLM query parser with catalog validation; hard filters and relaxation suggestions |
| `src/retrieve.py` · `src/personalize.py` | TF-IDF, BM25, dense, filter-first and hybrid retrieval; personalized ranking |
| `src/generate.py` | Grounded answers and the validator |
| `src/pipeline.py` · `src/agent.py` | End-to-end search; chat agent |
| `src/llm.py` · `src/store.py` | Shared LLM client and providers; store import, stock, webhooks, events |
| `src/evaluate.py` | Metrics, bootstrap CIs, LLM judge, experiments, report tables |
| `app/` | REST API, website widget, MCP server, Streamlit app |

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
11 ms. **Decision:** stream the ranked products first and the answer once it has passed the validator; unchecked text
is never shown.

**Separate production settings.** Ratings, hybrid retrieval and the optional re-ranker apply to the app only.
**Decision:** the app gets the improvements; every benchmark table keeps the evaluated configuration.

**Report negative results.** A parser rule, an adaptive personal weight and a cross-encoder re-ranker did not hold up
on test. **Decision:** reverted or switched off, and documented in the [final report](docs/final_report.md).

---

## 🧰 Built With

| Area | Tools |
|---|---|
| ML and retrieval | PyTorch (CPU), sentence-transformers (BGE-small), FAISS, scikit-learn, rank-bm25, SciPy |
| Data | pandas, NumPy, Pydantic |
| LLM | Google Gemini (default), OpenAI-compatible APIs, Anthropic Claude |
| Apps | FastAPI, Streamlit, MCP, Docker |
| Notebooks and tests | Jupyter, matplotlib, seaborn, pytest |

All versions are pinned in [requirements.txt](requirements.txt).

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

### Production extensions (each chosen on dev, reported once on test)

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

## 🌐 Live Demo

[AI Shopping Concierge App](https://ai-shopping-concierge.streamlit.app/) runs the same code as the full
app, showing the Appliances catalog with the **Chat**, **Shop** and **How it works** tabs.

| | Basic (no key, the default) | Full (the visitor's own key) |
|---|---|---|
| Finding and ranking products | ✅ same retrieval, filters and personalization | ✅ |
| Limits typed in the request ("under $50", "not stainless") | not read; price, category and brand can be set in *Narrow it down* | ✅ read by the LLM, enforced in Python |
| Answer | built from catalog fields | written by the LLM, checked by the grounding validator |
| Chat | each message is a plain search; Save / Similar / Buy work | full assistant: refine, compare, shortlist |
| Cost | none: no LLM call | billed to the visitor's own account |

A visitor's key (Gemini, OpenAI or Claude) stays in their browser session only: it is never saved, logged, cached or
shown, and no owner key is stored on the host. The store-owner tools are not part of the hosted demo.

---

## ☁️ Deployment

| Target | What runs | Where |
|---|---|---|
| Hosted demo | Streamlit app in demo mode (visitors bring their own AI key) | [ai-shopping-concierge.streamlit.app](https://ai-shopping-concierge.streamlit.app/) |
| Container image | REST API, website widget, webhooks, demo store page | Docker Hub: [`maazamjad110/ai-shopping-concierge`](https://hub.docker.com/r/maazamjad110/ai-shopping-concierge) |

```bash
docker pull maazamjad110/ai-shopping-concierge:latest
docker run -p 127.0.0.1:8000:8000 --env-file .env \
  -v ./data:/app/data -v ./artifacts:/app/artifacts -v ./stores:/app/stores \
  maazamjad110/ai-shopping-concierge:latest
```

The image runs as a non-root user and includes the embedding model; the catalog, embeddings, caches and imported
stores stay on mounted folders, and `.env` holds the LLM key. For a public server, put an HTTPS reverse proxy in
front and add the store's website to the allowed origins. Persistent stores and webhooks need a server with its own
disk.

---

## 🏪 Store Integration

The full app (run by a store owner) connects the assistant to a real shop without code:

| Source | Stock read from |
|---|---|
| Shopify product export (CSV) | inventory quantity, "continue selling", untracked |
| Shopify store address (`products.json`) | each variant's availability |
| WooCommerce product export (CSV) | "In stock?" and backorders |
| Any CSV / JSON list | an `in_stock` / `availability` column, or columns picked in the app |

- **Any layout:** a file the app does not recognise is mapped column by column (*which column is the price?*).
- **Website widget:** a two-line snippet ([below](#add-the-assistant-to-any-page)) adds a floating chat bubble or a
  search box to any page: Shopify, WooCommerce, WordPress or plain HTML.
- **Up to date:** a scheduled refresh (daily recommended, or *Update products now* in the app), or signed Shopify /
  WooCommerce webhooks within seconds. Only changed products are embedded again; webhooks are checked with the
  platform's HMAC signature, repeats are ignored, late or older updates never overwrite newer data, and deleted
  products cannot come back. The daily refresh stays as a safety net for missed deliveries.
- **Stock and ratings:** out-of-stock products are never recommended and unknown stock counts as available; optional
  rating columns are shown as stars and used in ranking.
- **Shopper events:** products shown, clicks, saves and Buy clicks feed the store's popularity (buy 5, save 3, click 1
  point; each browser counted once per product, action and day). No query text or IP address is stored, the random
  browser id is renewed every 30 days, and nothing is sent under Do Not Track. The owner sees a 30-day summary
  protected by a separate owner key.
- **AI agents:** an MCP server gives assistants such as Claude Desktop read-only `search_products` and `get_similar`;
  the app's *Connect to Claude Desktop* button sets it up.
- **REST API:** search and chat (with streaming: products first, then the checked answer), card actions, similar
  products, events and webhooks ([endpoints below](#-api-endpoints)).

### Add the assistant to any page

```html
<div id="concierge-widget"></div>
<script src="https://YOUR-API/widget.js" data-store="mystore"></script>
```

This shows a floating chat bubble. Add `data-mode="search"` for a search box instead, `data-key="..."` when API keys
are required, and `data-track="off"` to turn off shopper events. The store's website must be on the API's list of
allowed origins.

### 🤖 MCP server for AI agents

Read-only tools `search_products` and `get_similar`. The app's *Connect to Claude Desktop* button sets it up; by hand:

```json
{"mcpServers": {"shopping-assistant-mystore": {"command": "<project>/.venv/Scripts/python.exe",
  "args": ["<project>/app/mcp_server.py"], "env": {"CONCIERGE_STORE": "mystore"}}}}
```

### 🔌 API endpoints

Every endpoint takes an optional `store` (an imported store; leave it out for the demo catalog). The running API also
serves interactive documentation at `/docs`.

| Method & path | Purpose | LLM calls |
|---|---|---|
| `POST /search` · `POST /search/stream` | One search: understood limits, ranked products with score components, grounded answer | 2 |
| `POST /chat` · `POST /chat/stream` | One chat turn; the client sends back the returned `state` | 1–2 |
| `POST /chat/action` | Card buttons: save, similar, buy | 0 |
| `GET /similar/{product_id}` | Nearest products by embedding | 0 |
| `POST /events` · `GET /events/summary` | Shopper events · owner-only 30-day summary | 0 |
| `POST /webhooks/{store}/{shopify\|woocommerce}` | Signed product create, update and delete | 0 |
| `GET /meta` · `GET /health` · `GET /widget.js` · `GET /` | Store facts · health check · widget script · demo store page | 0 |

- **Streaming:** the `/stream` endpoints send newline-delimited JSON: a `products` event as soon as the products are
  ranked, then the answer once it has passed the grounding validator, so unchecked text is never sent.
- **Chat state:** the server keeps no conversation; the client sends the state back each turn and it is re-validated,
  including every limit.
- **Responses:** each search returns the parsed limits, the products with their score components (meaning, history,
  popularity, rating), the answer, flags (dropped filters, fallbacks) and per-stage timings.

---

## 🧪 Testing

323 automated tests cover data, retrieval, filters, parsing, ranking, grounded answers, evaluation metrics, the LLM
client and providers, the chat agent, stores and webhooks, the API, the MCP server and the app. Every test uses a fake
LLM on a toy catalog, so no test calls a real API. All reported results regenerate from cached outputs without new
LLM calls.

---

## 🛡 Error Handling & Security

| Layer | Situation | Behaviour |
|---|---|---|
| Parser | invalid JSON or unknown category / brand | one retry, then semantic-only; invalid values dropped and logged |
| Answer | invented ID, wrong price or brand | regenerate once, then a template answer from catalog fields |
| LLM | timeouts, 429, 503 | retries with backoff; per-process call cap; cached answers reused |
| Chat | unusable tool call, forged state | plain search fallback; client state re-validated every turn |
| API | abuse, cost | per-client rate limits, size limits, a cap on simultaneous LLM requests, optional API keys, CORS allow-list |
| Store links | server-side request forgery | addresses must resolve to public hosts, checked on every redirect; download cap |
| Webhooks | forged, replayed, out of order | HMAC check, duplicates ignored, older updates skipped, deletions remembered |
| Data | concurrent writers, crashes | lock file per store, atomic file swaps with rollback |
| Privacy | shopper data | no query text or IP in logs or events; random browser id renewed every 30 days; events kept one year |
| Hosted demo | visitors' keys | kept in the browser session only; never saved, logged, cached or shown |
| Untrusted text | product descriptions, chat state | treated as data in prompts; widget renders text only |

---

## ✅ Production Checklist

- [x] Hard limits enforced in code; answers grounded and validated
- [x] Caching, retries, call caps, rate limits, optional API keys
- [x] Signed webhooks with replay and ordering guard; scheduled refresh as safety net
- [x] Stock awareness; privacy-limited shopper events
- [x] Public hosted demo (visitors bring their own key)
- [x] Container image built and published on Docker Hub (non-root, model included)
- [ ] Run the container on a server behind HTTPS
- [ ] Shared database and vector index for several server copies
- [ ] User accounts, per-store keys entered in the app, owner dashboard
- [ ] Online A/B test with a holdout group

<div align="center">

**[⬆ Back to top](#-ai-shopping-concierge)**

</div>
