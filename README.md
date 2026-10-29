# Semantic Product Discovery & Shopping Concierge

Keyword search misses products described in different words, does not reliably enforce limits like
"under $80", and rarely explains its picks. This project builds and evaluates an end-to-end pipeline
over one Amazon product catalog:

```
query -> LLM query parsing (Gemini) -> catalog-validated hard filters (Python)
      -> BGE embeddings, filter-first retrieval -> personalized re-ranking
      -> grounded LLM answer + deterministic grounding validator
```

**Core rule:** the LLM *extracts* constraints; Python *enforces* them. The LLM never filters or ranks products.

## Status

Work in progress; deadline 30 Oct 2026. Design decisions and deviations will be summarized in the final report.

| Phase | State |
|---|---|
| 0 Plan ([intent](intent/intent.md)) | done |
| 1 Scaffold, shared LLM client, guardrails | done |
| 2 Data: `src/data.py`, [01_eda](notebooks/01_eda.ipynb) | done: 10,000 products, 24 categories, 24% without price |
| 3 Embeddings + baselines: `src/embed.py`, `src/retrieve.py`, [02_baselines](notebooks/02_baselines.ipynb) | done: S1 TF-IDF, S2 BM25, S3 dense; S4/S5 primitives |
| 4 Query parsing + hard filters: `src/parse.py`, `src/filters.py` | done: Gemini parser (prompt v2, tuned on dev), S4 post-filter and S5 filter-first |
| 5 Evaluation harness: `src/evaluate.py` | done: metrics with bootstrap CIs; S1-S5 run on all 25 queries; dev parse F1 0.80 (prompt v1) -> 1.00 (v2) |
| 6 Relevance labels: LLM judge + manual check | done: 659 pooled pairs judged (Gemini 3.8 Flash); 167 (25%) human-verified, agreement 81% exact / 92% relevant-vs-not |
| 7 Personalization: `src/personalize.py` | done: user profiles, weighted re-ranking, cold start; next-item test (3,995 users) HitRate@10 blend 0.063 vs popularity 0.028 vs random 0.002 |
| 8 Grounded answers: `src/generate.py`, `src/pipeline.py` | done: grounding validator + one regeneration + template fallback; 22/22 answers grounded (21 first try) |
| 9 Test-set evaluation: [04_evaluation](notebooks/04_evaluation.ipynb) | done |
| 10 [Final report](docs/final_report.md) | done |
| 11 Demo app, store connector (API + widget), MCP server: `app/`, `src/store.py` | done |
| 12 Shopping chat agent (`src/agent.py`), website chat bubble, any LLM provider (`src/llm.py`) | done |

## Systems compared (from the [technical spec](docs/tech_spec.md))

| ID | System |
|---|---|
| S1 | TF-IDF, raw query |
| S2 | BM25, raw query |
| S3 | Dense BGE + FAISS, no filters |
| S4 | Dense + post-filter (approved proposal design) |
| S5 | Dense + filter-first |
| S5x | S5 + exclusion terms ("not stainless"), reported separately |
| S6 | S5 + personalization |

Metrics: Precision@K, pooled Recall@K, NDCG@K, constraint adherence, parse F1, grounding rate, latency,
with 95% bootstrap confidence intervals.

## Results (15 held-out test queries; details in the [final report](docs/final_report.md))

| System | P@10 | NDCG@10 [95% CI] | Constraint adherence@10 |
|---|---|---|---|
| S1 TF-IDF | 0.46 | 0.40 [0.24, 0.58] | 0.43 |
| S2 BM25 | 0.51 | 0.51 [0.33, 0.70] | 0.44 |
| S3 dense | 0.67 | 0.65 [0.45, 0.82] | 0.63 |
| S4 post-filter | 0.79 | 0.74 [0.57, 0.89] | 1.00 |
| S5 filter-first | 0.79 | 0.74 [0.57, 0.89] | 1.00 |
| S5x + exclusions | 0.81 | 0.75 [0.59, 0.90] | 1.00 |
| S6 personalized | 0.79 | 0.71 [0.55, 0.87] | 1.00 |

- Query parsing F1 0.94 on test (target 0.85). Dense vs TF-IDF: +0.25 NDCG@10 [0.05, 0.48].
- Filter-first and post-filter return the same top 10 on every test query (a neutral result); filter-first is cheaper.
- Personalization: next-item HitRate@10 6.3% (blend) vs 2.8% (popularity); on query relevance it costs
  -0.026 NDCG@10 [-0.044, -0.007] while precision@10 is unchanged: it ranks some relevant products lower.
- Grounding: 22/22 answers grounded, 0 invented product IDs. Retrieval p95 11 ms; LLM generation p95 12.7 s (6 s
  end-to-end target not met on the free tier).

## Try it: demo app, chat assistant, store connector, MCP server

After setup (below), the concierge can be used in four ways, all read-only and built on the same `src/pipeline.py`:

**Demo app (Streamlit), no commands needed after starting it.** The *Chat* tab is a shopping assistant: ask for
products, then refine ("cheaper", "only GE", "not stainless"), compare ("compare 1 and 2"), find similar items, keep
a shortlist, and get the product page to buy (no orders are placed). The *Shop* tab is single-shot search with the
enforced limits, score components and the grounded answer. The *Add your store* tab imports a store, turns the website
connection on and off, and connects Claude Desktop, all with buttons.

```bash
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

**Plug into an existing store.** Import the store's product export (CSV or JSON, a file or a URL), start the API, and
paste two lines into any page:

```bash
.venv/Scripts/python.exe -m src.store --name mystore --file products_export.csv --preset shopify
.venv/Scripts/python.exe -m src.store --name mystore --file https://mystore.com/products.json --preset shopify_json
.venv/Scripts/python.exe -m src.store --name mystore --file feed.csv --map title=Name,price=Cost   # any other format
.venv/Scripts/python.exe -m uvicorn app.api:app --port 8000      # docs at http://localhost:8000/docs
```

```html
<div id="concierge-widget"></div>
<script src="http://localhost:8000/widget.js" data-store="mystore"></script>
```

This adds a floating chat bubble (the shopping assistant); add `data-mode="search"` for a plain search box instead.

Presets: `shopify` (admin CSV export), `shopify_json` (`/products.json`), `woocommerce` (CSV export), `generic`
(`product_id,title,brand,category,price,description,features`). API: `POST /search`, `POST /chat` (one chat turn; the
client sends back the returned state), `POST /chat/action` (card buttons, no LLM call), `GET /similar/{id}`,
`GET /meta`, `GET /health`; `http://localhost:8000/` shows a demo store page with the widget. Imported stores live in
`stores/` (not committed); their shoppers have no history, so personalization is off for them. Browsers may call the
API only from origins listed in `config.yaml` (`app.allowed_origins`) or `CONCIERGE_ALLOWED_ORIGINS`
(comma-separated), so add your store's domain there. Each server process makes at most `app.max_llm_calls` Gemini
calls (default 300), then answers from catalog fields only. The API listens on localhost; put it behind a proxy with
authentication before exposing it.

**MCP server for AI shopping agents.** Tools `search_products` and `get_similar` (read-only). The app's *Connect to
Claude Desktop* button sets it up; by hand, add:

```json
{"mcpServers": {"shopping-assistant-mystore": {"command": "<project>/.venv/Scripts/python.exe",
  "args": ["<project>/app/mcp_server.py"], "env": {"CONCIERGE_STORE": "mystore"}}}}
```

**Any LLM provider.** Gemini is the default (and the reported results are Gemini results). Set `llm.provider` in
`config.yaml` to `openai` for ChatGPT, to `openai` with a `base_url` for open-weight models served by Ollama, vLLM,
LM Studio and similar OpenAI-compatible servers, or to `anthropic` for Claude, and set `llm.model` / `llm.judge_model`.
Keys go in `.env` as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (a local open-weight server needs none). Caching, call
caps, retries, structured output and the chat agent's function calling work the same for every provider.

## Setup

Requires Python 3.12 (created with [uv](https://docs.astral.sh/uv/)) and a Gemini API key.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Linux/macOS: .venv/bin/python
cp .env.example .env    # then put your key after GEMINI_API_KEY=
.venv/Scripts/python.exe -m pytest -q
```

Data is downloaded from Hugging Face into `data/raw/` (not committed), then processed:

```bash
B=https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main
curl -L -o data/raw/meta_Appliances.jsonl "$B/raw/meta_categories/meta_Appliances.jsonl"          # 285 MB
curl -L -o data/raw/Appliances_rating_only.csv "$B/benchmark/0core/rating_only/Appliances.csv"   # 122 MB
.venv/Scripts/python.exe -m src.data     # -> data/processed/, eval/results/cleaning_log.json
.venv/Scripts/python.exe -m src.embed    # -> artifacts/embeddings.npy (~27 min on CPU, minutes on a GPU)
.venv/Scripts/python.exe -m src.parse --split dev    # Gemini, cached; also --split test
.venv/Scripts/python.exe -m src.evaluate --split dev # S1-S5 runs, no LLM calls -> eval/results/runs.csv
.venv/Scripts/python.exe -m src.evaluate judge       # LLM relevance judge (cached) -> eval/results/judge_labels.csv
.venv/Scripts/python.exe -m src.evaluate qrels       # merge with human labels -> eval/qrels.csv
.venv/Scripts/python.exe -m src.personalize          # next-item experiment, no LLM -> eval/results/exp_b_*.csv
.venv/Scripts/python.exe -m src.pipeline             # grounded answers for all queries -> eval/results/generation.csv
.venv/Scripts/python.exe -m src.evaluate judge-s6    # personalized runs + judge their new items -> eval/results/runs_s6.csv
.venv/Scripts/python.exe -m src.evaluate qrels       # merge judge + human labels -> eval/qrels.csv
.venv/Scripts/python.exe -m src.evaluate report --split test  # all test tables -> eval/results/test_*.csv
```

## Repository layout

```
docs/        requirements, technical specification, approved proposal
intent/      intent.md (playbook Stage 1)
src/         pipeline modules (llm.py = cached, rate-limited Gemini client; store.py = feed importer)
app/         Streamlit demo, REST API + drop-in widget, MCP server
tests/       pytest suite (never calls the real API)
notebooks/   EDA, baselines, pipeline demo, evaluation
eval/        benchmark queries, relevance labels, results
config.yaml  all settings (seed 42, models, weights)
```

## Reproducibility

Seed 42, pinned versions, temperature 0, and every LLM call cached by
(model, prompt version, schema, settings, prompt). Reported numbers come from files in `eval/results/`.

## Data and credits

Amazon Reviews 2023 (McAuley Lab): Hou et al., *Bridging Language and Items for Retrieval and
Recommendation*, 2024. Used for academic, non-commercial purposes. Embeddings:
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). LLM: Google Gemini API.
