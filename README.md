# Semantic Product Discovery & Shopping Concierge

atomcamp AI Bootcamp final project (NLP/LLM track) by Maaz Amjad.

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

Work in progress; deadline 28 Oct 2026. Design decisions and deviations will be summarized in the final report.

| Phase | State |
|---|---|
| 0 Plan ([intent](intent/intent.md)) | done |
| 1 Scaffold, shared LLM client, guardrails | done |
| 2 Data (Amazon Reviews 2023, Appliances) | dataset confirmed; `data.py` next |
| 3-10 Embeddings, parsing, evaluation, personalization, generation, report | planned |

## Systems compared (from the [technical spec](docs/tech_spec.md))

| ID | System |
|---|---|
| S1 | TF-IDF, raw query |
| S2 | BM25, raw query |
| S3 | Dense BGE + FAISS, no filters |
| S4 | Dense + post-filter (approved proposal design) |
| S5 | Dense + filter-first |
| S6 | S5 + personalization |

Metrics: Precision@K, pooled Recall@K, NDCG@K, constraint adherence, parse F1, grounding rate, latency,
with 95% bootstrap confidence intervals. Results will be added here when available.

## Setup

Requires Python 3.12 (created with [uv](https://docs.astral.sh/uv/)) and a Gemini API key.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Linux/macOS: .venv/bin/python
cp .env.example .env    # then put your key after GEMINI_API_KEY=
.venv/Scripts/python.exe -m pytest -q
```

Data is downloaded from Hugging Face into `data/raw/` (not committed):
`raw/meta_categories/meta_Appliances.jsonl` and `benchmark/0core/rating_only/Appliances.csv`.

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
