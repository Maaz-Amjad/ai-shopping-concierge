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

Work in progress; deadline 29 Oct 2026. Design decisions and deviations will be summarized in the final report.

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
| 9 Test-set evaluation: [04_evaluation](notebooks/04_evaluation.ipynb) | done (S6 provisional: 11% of its top 10 not yet labelled) |
| 10 [Final report](docs/final_report.md) | draft complete; S6 numbers refreshed once labelling finishes |

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
| S1 TF-IDF | 0.46 | 0.41 [0.24, 0.59] | 0.43 |
| S2 BM25 | 0.51 | 0.52 [0.33, 0.71] | 0.44 |
| S3 dense | 0.67 | 0.66 [0.46, 0.83] | 0.63 |
| S4 post-filter | 0.79 | 0.75 [0.59, 0.89] | 1.00 |
| S5 filter-first | 0.79 | 0.75 [0.59, 0.89] | 1.00 |
| S5x + exclusions | 0.81 | 0.76 [0.59, 0.91] | 1.00 |
| S6 personalized (provisional) | 0.71 | 0.66 [0.52, 0.81] | 1.00 |

- Query parsing F1 0.94 on test (target 0.85). Dense vs TF-IDF: +0.25 NDCG@10 [0.05, 0.47].
- Filter-first and post-filter return the same top 10 on every test query (a neutral result); filter-first is cheaper.
- Personalization: next-item HitRate@10 6.3% (blend) vs 2.8% (popularity); on query relevance it costs
  -0.025 NDCG@10 [-0.052, -0.004] on judged items (S6's row above counts its unlabelled items as irrelevant).
- Grounding: 22/22 answers grounded, 0 invented product IDs. Retrieval p95 11 ms; LLM generation p95 12.7 s (6 s
  end-to-end target not met on the free tier).

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
```

## Repository layout

```
docs/        requirements, technical specification, approved proposal
intent/      intent.md (playbook Stage 1)
src/         pipeline modules (llm.py = cached, rate-limited Gemini client)
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
