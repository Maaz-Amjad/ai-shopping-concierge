"""End-to-end orchestration (tech spec 2 and 11): parse -> filter-first retrieval -> rank -> grounded answer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src import ROOT, load_config, set_seed
from src.filters import relaxation_suggestions
from src.generate import MAX_WORDS, GenerationResult, answer_text, generate_answer, unpriced_section
from src.llm import LLMClient
from src.parse import CatalogMeta, ParsedQuery, parse_query, validate
from src.personalize import Weights, build_profiles, rank
from src.retrieve import DenseIndex, retrieve


@dataclass
class Resources:
    """Everything loaded once and shared by all queries."""

    cfg: dict
    catalog: pd.DataFrame
    embeddings: np.ndarray
    index: DenseIndex
    model: Any
    meta: CatalogMeta
    profiles: dict[str, np.ndarray]
    client: LLMClient
    brands: list[str]


def load_resources(cfg: dict | None = None, max_calls: int | None = None, store: str | None = None) -> Resources:
    """Load the benchmark catalog, or an imported store (`python -m src.store`) whose users are all cold-start."""
    from src.data import load_catalog, load_interactions
    from src.embed import load_embeddings, load_model

    cfg = cfg or load_config()
    set_seed(cfg["seed"])
    if store is not None:
        from src.store import STORES_DIR, check_store_name

        check_store_name(store)                    # never a path: names are plain identifiers
        store_dir = STORES_DIR / store
        catalog = pd.read_pickle(store_dir / "catalog.pkl")
        embeddings, product_ids = load_embeddings(store_dir)
    else:
        catalog = load_catalog()
        embeddings, product_ids = load_embeddings()
    if not (product_ids == catalog["product_id"].to_numpy()).all():
        raise ValueError("embeddings are not aligned with the catalog; rebuild with python -m src.embed")
    llm = cfg["llm"]
    client = LLMClient(llm["model"], ROOT / llm["cache_path"], ROOT / llm["call_log_path"], rpm=llm["rpm"],
                       temperature=llm["temperature"], max_output_tokens=llm["max_output_tokens"],
                       max_calls=max_calls)
    brands = sorted(catalog["brand"].dropna().astype(str).unique().tolist())
    profiles = {} if store is not None else build_profiles(load_interactions(), catalog, embeddings)
    return Resources(cfg, catalog, embeddings, DenseIndex(embeddings), load_model(cfg["embedding"]["model"]),
                     CatalogMeta.from_catalog(catalog), profiles, client, brands)


@dataclass
class PipelineResult:
    parsed_query: ParsedQuery
    candidates: pd.DataFrame                 # filter-first pool (row, score, rank)
    ranked: pd.DataFrame                     # top-n with score components
    answer: GenerationResult
    timings_ms: dict[str, float]
    flags: dict[str, Any] = field(default_factory=dict)   # dropped filters, fallbacks, price exclusions


def respond(query: str, res: Resources, user_id: str | None = None, n: int | None = None,
            parsed: ParsedQuery | None = None, use_exclusions: bool = False) -> PipelineResult:
    """Answer one query (spec 11). Pass `parsed` to reuse a cached parse; unknown users are cold-start.

    use_exclusions=False matches S6 = S5 + personalization; True adds the optional exclusions (spec 8.3).
    """
    from src.embed import encode_query, use_query_instruction

    cfg, timings = res.cfg, {}
    n = n or cfg["generation"]["top_n"]
    flags: dict[str, Any] = {}
    t = time.perf_counter()
    if parsed is None:
        pr = parse_query(query, res.meta, res.client, raise_on_quota=False)
        parsed = pr.parsed
        flags.update(parse_log=pr.log, parser_fallback=pr.fallback)
    timings["parse"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    qvec = encode_query(parsed.semantic_query, res.model, use_instruction=use_query_instruction(cfg))
    r = cfg["retrieval"]
    result = retrieve(parsed, "filter_first", r["pool_size"], catalog=res.catalog, index=res.index, query_vec=qvec,
                      n=n, k0=r["k0"], k_max=r["k_max"], use_exclusions=use_exclusions)
    timings["retrieve"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    user_vec = res.profiles.get(user_id) if user_id else None
    ranked = rank(result.candidates, user_vec, Weights.from_config(cfg), catalog=res.catalog,
                  embeddings=res.embeddings).head(n)
    timings["rank"] = (time.perf_counter() - t) * 1000
    flags.update(cold_start=user_vec is None, n_satisfying=result.n_satisfying,
                 price_excluded=result.price_excluded, n_unpriced=len(result.unpriced))

    t = time.perf_counter()
    items = res.catalog.iloc[ranked["row"].to_numpy()].reset_index(drop=True)
    suggestions = relaxation_suggestions(res.catalog, parsed) if items.empty else None
    answer = generate_answer(query, parsed, items, res.client, res.brands, n_requested=n, suggestions=suggestions)
    if not result.unpriced.empty:
        answer.text += "\n" + unpriced_section(res.catalog.iloc[result.unpriced["row"].to_numpy()])
    timings["generate"] = (time.perf_counter() - t) * 1000
    timings["total"] = sum(timings.values())
    flags.update(generation_fallback=answer.fallback, grounding_errors=answer.errors)
    return PipelineResult(parsed, result.candidates, ranked, answer, timings, flags)


PRODUCT_FIELDS = ["product_id", "title", "brand", "category", "price"]
SCORE_FIELDS = ["rank", "final", "s_sem", "s_pers", "s_pop", "cosine"]


def search_products(query: str, res: Resources, *, max_price: float | None = None, min_price: float | None = None,
                    category: str | None = None, brand: str | None = None, n: int | None = None,
                    user_id: str | None = None, use_exclusions: bool = True) -> PipelineResult:
    """Tool interface (spec 11): the LLM parses the query; filters passed explicitly (a store's UI or an AI agent)
    override the parsed ones and are validated like parser output (unknown category/brand dropped and logged).

    "not X" exclusions are applied by default (S5x, spec 8.3), since users see them as understood limits.
    """
    explicit = {k: v for k, v in {"max_price": max_price, "min_price": min_price, "category": category,
                                  "brand": brand}.items() if v is not None}
    if not explicit:
        return respond(query, res, user_id=user_id, n=n, use_exclusions=use_exclusions)
    pr = parse_query(query, res.meta, res.client, raise_on_quota=False)
    parsed, log = validate(pr.parsed.model_copy(update=explicit), query, res.meta)
    out = respond(query, res, user_id=user_id, n=n, parsed=parsed, use_exclusions=use_exclusions)
    out.flags.update(parse_log=pr.log + [f"explicit filter: {k}={v}" for k, v in explicit.items()] + log,
                     parser_fallback=pr.fallback)
    return out


def get_similar(product_id: str, res: Resources, n: int = 5) -> pd.DataFrame:
    """Products closest to `product_id` by embedding cosine (spec 11 get_similar); raises KeyError if unknown."""
    rows = np.flatnonzero(res.catalog["product_id"].to_numpy() == product_id)
    if len(rows) == 0:
        raise KeyError(f"unknown product_id {product_id}")
    scores = res.embeddings @ res.embeddings[rows[0]]
    scores[rows[0]] = -np.inf
    top = np.argsort(-scores, kind="stable")[:n]
    out = res.catalog.iloc[top][PRODUCT_FIELDS].reset_index(drop=True)
    return out.assign(similarity=scores[top].astype(float).round(4))


def _plain(value: Any) -> Any:
    """JSON-safe scalars: numpy numbers to Python, NaN to None."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: _plain(v) for k, v in row.items()} for row in df.to_dict("records")]


def result_to_dict(query: str, out: PipelineResult, res: Resources) -> dict[str, Any]:
    """JSON view of a pipeline result: parsed filters, products with score components (NFR-7), answer, flags."""
    products = res.catalog.iloc[out.ranked["row"].to_numpy()][PRODUCT_FIELDS].reset_index(drop=True)
    scores = out.ranked[[c for c in SCORE_FIELDS if c in out.ranked]].reset_index(drop=True)
    reasons = {r.product_id: r.reason for r in out.answer.answer.recommendations} if out.answer.answer else {}
    items = records(pd.concat([products, scores], axis=1))
    for item in items:
        item["reason"] = reasons.get(item["product_id"])
    flags = {k: (v if isinstance(v, (list, str, bool, type(None))) else _plain(v)) for k, v in out.flags.items()}
    return {"query": query, "parsed": out.parsed_query.model_dump(), "products": items, "answer": out.answer.text,
            "flags": flags, "timings_ms": {k: round(v, 1) for k, v in out.timings_ms.items()}}


def word_count(text: str) -> int:
    return len(text.split())


def generation_benchmark(limit: int | None, cfg: dict | None = None) -> pd.DataFrame:
    """Spec 10.7: run respond() on every benchmark query (cached parses, new user) and save per-query rows."""
    import json

    from src.evaluate import load_parsed, load_queries, to_parsed_query

    res = load_resources(cfg, max_calls=limit)
    parsed = load_parsed().set_index("query_id")
    rows = []
    for _, q in load_queries().iterrows():
        out = respond(q["text"], res, parsed=to_parsed_query(parsed.loc[q["query_id"]]))
        a = out.answer
        first = a.errors                  # errors of every rejected attempt
        rows.append({
            "query_id": q["query_id"], "split": q["split"], "n_returned": len(out.ranked),
            "attempts": a.attempts, "fallback": a.fallback or "",
            "grounded_first_try": a.attempts == 1 and a.fallback is None,
            "grounded_final": a.fallback != "template",
            "bad_ids": sum(e.startswith("unknown product id") for e in first),
            "bad_prices": sum(e.startswith("price $") for e in first),
            "bad_brands": sum(e.startswith("brand ") for e in first),
            "invalid_json": sum(e.startswith("invalid JSON") for e in first),
            "llm_errors": sum(e.startswith("LLM error") for e in first),
            "answer_words": word_count(a.text), "over_max_words": word_count(a.text) > MAX_WORDS,
            "llm_words": word_count(answer_text(a.answer)) if a.answer else 0, "errors": json.dumps(a.errors),
            "ranked_ids": json.dumps(out.ranked["product_id"].tolist()),
            "recommended_ids": json.dumps([r.product_id for r in a.answer.recommendations] if a.answer else []),
            "answer": a.text, **{f"{k}_ms": round(v, 1) for k, v in out.timings_ms.items()},
        })
    df = pd.DataFrame(rows)
    out_dir = ROOT / "eval" / "results"
    df.to_csv(out_dir / "generation.csv", index=False)
    llm_rows = df[df["fallback"] != "no_result"]
    print(f"generated {len(df)} answers ({len(llm_rows)} via LLM, {int((df.fallback == 'no_result').sum())} "
          f"no-result) | API calls this run: {res.client.calls_made} | cache hits: {res.client.cache_hits}")
    print(f"grounded first try: {int(llm_rows.grounded_first_try.sum())}/{len(llm_rows)} | final (no template): "
          f"{int(llm_rows.grounded_final.sum())}/{len(llm_rows)} | rejected attempts: invented ids "
          f"{int(df.bad_ids.sum())}, price mismatches {int(df.bad_prices.sum())}, brand mismatches "
          f"{int(df.bad_brands.sum())}, invalid JSON {int(df.invalid_json.sum())}, API errors {int(df.llm_errors.sum())}")
    print(f"answer words: median {df.answer_words.median():.0f}, max {df.answer_words.max()}, over {MAX_WORDS}: "
          f"{int(df.over_max_words.sum())} | LLM-written words max {df.llm_words.max()} | "
          f"generate ms median {df.generate_ms.median():.0f}, p95 {df.generate_ms.quantile(0.95):.0f}")
    return df


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generation benchmark over all queries (spec 10.7).")
    parser.add_argument("--limit", type=int, default=None, help="max real API calls (dry-run budget)")
    generation_benchmark(parser.parse_args(argv).limit)


if __name__ == "__main__":
    main()
