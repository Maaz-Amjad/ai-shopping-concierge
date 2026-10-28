"""End-to-end orchestration (tech spec 2 and 11): parse -> filter-first retrieval -> rank -> grounded answer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src import ROOT, load_config, set_seed
from src.filters import relaxation_suggestions
from src.generate import GenerationResult, generate_answer
from src.llm import LLMClient
from src.parse import CatalogMeta, ParsedQuery, parse_query
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


def load_resources(cfg: dict | None = None, max_calls: int | None = None) -> Resources:
    from src.data import load_catalog, load_interactions
    from src.embed import load_embeddings, load_model

    cfg = cfg or load_config()
    set_seed(cfg["seed"])
    catalog = load_catalog()
    embeddings, product_ids = load_embeddings()
    if not (product_ids == catalog["product_id"].to_numpy()).all():
        raise ValueError("embeddings are not aligned with the catalog; rebuild with python -m src.embed")
    llm = cfg["llm"]
    client = LLMClient(llm["model"], ROOT / llm["cache_path"], ROOT / llm["call_log_path"], rpm=llm["rpm"],
                       temperature=llm["temperature"], max_output_tokens=llm["max_output_tokens"],
                       max_calls=max_calls)
    brands = sorted(catalog["brand"].dropna().astype(str).unique().tolist())
    return Resources(cfg, catalog, embeddings, DenseIndex(embeddings), load_model(cfg["embedding"]["model"]),
                     CatalogMeta.from_catalog(catalog), build_profiles(load_interactions(), catalog, embeddings),
                     client, brands)


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
    timings["generate"] = (time.perf_counter() - t) * 1000
    timings["total"] = sum(timings.values())
    flags.update(generation_fallback=answer.fallback, grounding_errors=answer.errors)
    return PipelineResult(parsed, result.candidates, ranked, answer, timings, flags)
