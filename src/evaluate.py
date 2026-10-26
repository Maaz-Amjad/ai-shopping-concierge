"""Evaluation: metrics, bootstrap CIs, parse F1, pooling, and the S1-S5 run harness (spec section 10).

Run `python -m src.evaluate --split dev` to run S1-S5 on all benchmark queries (no LLM calls: saved
parses are used), save eval/results/runs.csv + run_info.csv, and print summaries for one split.

Conventions:
- Relevant = grade >= 1. Unjudged items count as grade 0.
- Recall@K and NDCG@K are NaN when the pool holds no relevant item; NaN queries are left out of means.
- Constraint checks are written independently of src.filters so the evaluation does not reuse the
  code it evaluates. A null price violates any price constraint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src import ROOT, load_config, set_seed
from src.parse import ParsedQuery

RESULTS_DIR = ROOT / "eval" / "results"
# S4/S5 apply hard filters only (spec 10.2); S5x = S5 + exclusion terms, reported separately (spec 8.3).
SYSTEMS = ("S1", "S2", "S3", "S4", "S5", "S5x")
HARD_FIELDS = ("max_price", "min_price", "category", "brand")
PRICE_TOL = 0.01


# ---- ranking metrics (one query; grades listed in rank order) ------------------------------


def precision_at_k(grades: Sequence[float], k: int) -> float:
    """Relevant items (grade >= 1) in the top k, divided by k (a short list still divides by k)."""
    return float(sum(g >= 1 for g in list(grades)[:k])) / k


def recall_at_k(grades: Sequence[float], n_relevant_pool: int, k: int) -> float:
    """Relevant items in the top k divided by all relevant items in the pool; NaN if the pool has none."""
    if n_relevant_pool == 0:
        return float("nan")
    return float(sum(g >= 1 for g in list(grades)[:k])) / n_relevant_pool


def dcg(grades: Sequence[float], k: int) -> float:
    """Sum of (2^grade - 1) / log2(rank + 1) over the top k."""
    g = np.asarray(list(grades)[:k], dtype=float)
    return float(((2 ** g - 1) / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg_at_k(grades: Sequence[float], pool_grades: Sequence[float], k: int) -> float:
    """DCG of the ranking divided by the DCG of the ideal ordering of all pooled grades; NaN if ideal is 0."""
    ideal = dcg(sorted(pool_grades, reverse=True), k)
    return float("nan") if ideal == 0 else dcg(grades, k) / ideal


# ---- constraint adherence --------------------------------------------------------------------


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def has_hard_gold(gold: pd.Series) -> bool:
    return any(not _blank(gold[f"gold_{f}"]) for f in HARD_FIELDS)


def satisfies_gold(item: pd.Series, gold: pd.Series) -> bool:
    """True if one catalog item meets every gold hard constraint (price, category, brand)."""
    price = item["price"]
    if not _blank(gold["gold_max_price"]) and (_blank(price) or price > float(gold["gold_max_price"])):
        return False
    if not _blank(gold["gold_min_price"]) and (_blank(price) or price < float(gold["gold_min_price"])):
        return False
    if not _blank(gold["gold_category"]) and item["category"] != gold["gold_category"]:
        return False
    if not _blank(gold["gold_brand"]) and \
            (_blank(item["brand"]) or str(item["brand"]).casefold() != str(gold["gold_brand"]).casefold()):
        return False
    return True


def violates_exclusion(item: pd.Series, gold: pd.Series) -> bool:
    """True if title or features contain the gold excluded term (reported separately, spec 8.3)."""
    if _blank(gold["gold_exclude"]):
        return False
    text = " ".join("" if _blank(item[c]) else str(item[c]) for c in ("title", "features")).lower()
    return str(gold["gold_exclude"]).lower() in text


def constraint_adherence(items: pd.DataFrame, gold: pd.Series, k: int) -> float:
    """Share of the top-k items meeting every gold hard constraint; NaN when nothing was returned."""
    top = items.head(k)
    if top.empty:
        return float("nan")
    return float(np.mean([satisfies_gold(row, gold) for _, row in top.iterrows()]))


# ---- parse F1 ----------------------------------------------------------------------------------


def _field_values(field: str, value: object) -> set:
    """A field as a set of comparable items (empty when not set)."""
    if field == "exclude_terms":
        if isinstance(value, (list, tuple)):
            return {str(t).strip().lower() for t in value if str(t).strip()}
        return set() if _blank(value) else {str(value).strip().lower()}
    if _blank(value):
        return set()
    if field in ("max_price", "min_price"):
        return {round(float(value) / PRICE_TOL)}          # compare prices to the cent
    return {str(value).strip().casefold()}


def parse_counts(pred: pd.DataFrame, gold: pd.DataFrame,
                 fields: Iterable[str] = (*HARD_FIELDS, "exclude_terms")) -> pd.DataFrame:
    """Per-field TP/FP/FN over queries. A wrong value counts as one FP and one FN.

    `pred` has query_id + ParsedQuery columns; `gold` has query_id + gold_<field> columns
    (gold_exclude for exclude_terms).
    """
    merged = gold.merge(pred, on="query_id", how="left")
    rows = []
    for field in fields:
        gold_col = "gold_exclude" if field == "exclude_terms" else f"gold_{field}"
        tp = fp = fn = 0
        for _, r in merged.iterrows():
            p, g = _field_values(field, r[field]), _field_values(field, r[gold_col])
            tp, fp, fn = tp + len(p & g), fp + len(p - g), fn + len(g - p)
        rows.append({"field": field, "tp": tp, "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


def micro_f1(counts: pd.DataFrame) -> dict[str, float]:
    """Micro precision, recall and F1 from summed TP/FP/FN. Undefined ratios count as 1.0 (nothing to get wrong)."""
    tp, fp, fn = (int(counts[c].sum()) for c in ("tp", "fp", "fn"))
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---- statistics --------------------------------------------------------------------------------


def bootstrap_ci(values: Sequence[float], n_resamples: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile 95% CI over queries (NaN values dropped first)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_resamples, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = 1000,
                     seed: int = 42) -> tuple[float, float, float]:
    """Mean of (b - a) and its 95% CI, resampling queries jointly; pairs with a NaN are dropped."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return bootstrap_ci(b[keep] - a[keep], n_resamples, seed)


def latency_summary(ms: Sequence[float]) -> dict[str, float]:
    """Median and p95 of latencies in milliseconds."""
    v = np.asarray(ms, dtype=float)
    return {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))}


# ---- pooling and per-query metrics against qrels --------------------------------------------------


def build_pool(runs: pd.DataFrame, depth: int = 10) -> pd.DataFrame:
    """Union of the top-`depth` products of every system, per query (spec 10.3)."""
    top = runs[runs["rank"] <= depth]
    return top[["query_id", "product_id"]].drop_duplicates().sort_values(["query_id", "product_id"]) \
        .reset_index(drop=True)


def ranking_metrics(runs: pd.DataFrame, qrels: pd.DataFrame, ks: Iterable[int] = (5, 10)) -> pd.DataFrame:
    """Per (system, query) P@K, R@K and NDCG@K. Unjudged products get grade 0.

    `runs`: system, query_id, rank, product_id. `qrels`: query_id, product_id, grade.
    Queries with no run rows for a system get an empty ranking.
    """
    grade_of = {(q, p): g for q, p, g in qrels[["query_id", "product_id", "grade"]].itertuples(index=False)}
    pool_grades = qrels.groupby("query_id")["grade"].apply(list).to_dict()
    rows = []
    for system in runs["system"].unique():
        sys_runs = runs[runs["system"] == system]
        for qid in sorted(set(qrels["query_id"]) | set(sys_runs["query_id"])):
            ranked = sys_runs[sys_runs["query_id"] == qid].sort_values("rank")["product_id"]
            grades = [grade_of.get((qid, p), 0) for p in ranked]
            pool = pool_grades.get(qid, [])
            n_rel = sum(g >= 1 for g in pool)
            row = {"system": system, "query_id": qid}
            for k in ks:
                row[f"P@{k}"] = precision_at_k(grades, k)
                row[f"R@{k}"] = recall_at_k(grades, n_rel, k)
                row[f"NDCG@{k}"] = ndcg_at_k(grades, pool, k)
            rows.append(row)
    return pd.DataFrame(rows)


# ---- run harness ----------------------------------------------------------------------------------


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def load_queries() -> pd.DataFrame:
    return pd.read_csv(ROOT / "eval" / "queries.csv")


def load_parsed(splits: Iterable[str] = ("dev", "test")) -> pd.DataFrame:
    """Saved parser outputs (no LLM calls)."""
    frames = [pd.read_json(RESULTS_DIR / f"parsed_queries_{s}.jsonl", lines=True) for s in splits]
    return pd.concat(frames, ignore_index=True)


def to_parsed_query(row: pd.Series) -> ParsedQuery:
    fields = ParsedQuery.model_fields
    return ParsedQuery(**{f: (None if f != "exclude_terms" and _blank(row[f]) else row[f]) for f in fields})


def run_systems(queries: pd.DataFrame, parsed: pd.DataFrame, cfg: dict,
                systems: Sequence[str] = SYSTEMS, depth: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run S1-S5 (+ S5x) on every query. Returns (runs: top-`depth` per system/query, info: one row per system/query).

    S1-S3 use the raw query text; S4/S5 use the parsed semantic_query and hard filters; S5x adds exclusions.
    S4 is called with n=depth so post-filter keeps widening K until `depth` items survive (or k_max).
    """
    from src.data import load_catalog
    from src.embed import encode_query, load_embeddings, load_model, use_query_instruction
    from src.retrieve import BM25Retriever, DenseIndex, TfidfRetriever, retrieve

    set_seed(cfg["seed"])
    catalog = load_catalog()
    embeddings, product_ids = load_embeddings()
    if not (product_ids == catalog["product_id"].to_numpy()).all():
        raise ValueError("embeddings are not aligned with the catalog; rebuild with python -m src.embed")
    texts = catalog["product_text"].tolist()
    lexical = {"S1": TfidfRetriever(texts), "S2": BM25Retriever(texts)}
    index = DenseIndex(embeddings)
    model = load_model(cfg["embedding"]["model"])
    prefix = use_query_instruction(cfg)
    r = cfg["retrieval"]
    parsed_by_id = parsed.set_index("query_id")

    run_rows, info_rows = [], []
    for _, q in queries.iterrows():
        qid, pq = q["query_id"], to_parsed_query(parsed_by_id.loc[q["query_id"]])
        t = time.perf_counter()
        raw_vec = encode_query(q["text"], model, use_instruction=prefix)
        raw_encode_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        sem_vec = encode_query(pq.semantic_query, model, use_instruction=prefix)
        sem_encode_ms = (time.perf_counter() - t) * 1000
        for system in systems:
            info = {"query_id": qid, "split": q["split"], "system": system, "k_used": None,
                    "n_satisfying": None, "price_excluded": None, "n_unpriced": 0}
            t = time.perf_counter()
            if system in lexical:
                cands, encode_ms = lexical[system].search(q["text"], depth), 0.0
            elif system == "S3":
                # Plain FAISS search: retrieve(mode="dense") would also time mask building it never uses.
                cands, encode_ms = index.search(raw_vec, r["pool_size"]), raw_encode_ms
            else:
                mode, exclusions = {"S4": ("post_filter", False), "S5": ("filter_first", False),
                                    "S5x": ("filter_first", True)}[system]
                encode_ms = sem_encode_ms
                res = retrieve(pq, mode, r["pool_size"], catalog=catalog, index=index, query_vec=sem_vec,
                               n=depth, k0=r["k0"], k_max=r["k_max"], use_exclusions=exclusions)
                cands = res.candidates
                info.update(k_used=res.k_used, n_satisfying=res.n_satisfying,
                            price_excluded=res.price_excluded, n_unpriced=len(res.unpriced))
            info.update(search_ms=(time.perf_counter() - t) * 1000, encode_ms=encode_ms,
                        pool_len=len(cands), n_returned=min(len(cands), depth))
            info_rows.append(info)
            for c in cands.head(depth).itertuples(index=False):
                run_rows.append({"query_id": qid, "system": system, "rank": int(c.rank), "row": int(c.row),
                                 "product_id": catalog["product_id"].iat[int(c.row)], "score": float(c.score)})
    tag = config_hash(cfg)
    return pd.DataFrame(run_rows).assign(config_hash=tag), pd.DataFrame(info_rows).assign(config_hash=tag)


def adherence_table(runs: pd.DataFrame, queries: pd.DataFrame, catalog: pd.DataFrame,
                    ks: Iterable[int] = (5, 10)) -> pd.DataFrame:
    """Per (system, query) hard-constraint adherence@K and exclusion violations@10, for queries with gold limits."""
    gold_by_id = queries.set_index("query_id")
    rows = []
    for (system, qid), group in runs.groupby(["system", "query_id"]):
        gold = gold_by_id.loc[qid]
        items = catalog.iloc[group.sort_values("rank")["row"].to_numpy()]
        row = {"system": system, "query_id": qid}
        for k in ks:
            row[f"adherence@{k}"] = constraint_adherence(items, gold, k) if has_hard_gold(gold) else float("nan")
        row["excl_violations@10"] = sum(violates_exclusion(it, gold) for _, it in items.head(10).iterrows()) \
            if not _blank(gold["gold_exclude"]) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run S1-S5 on the benchmark and print summaries.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="split whose adherence and parse F1 are printed (tune on dev, report test)")
    args = parser.parse_args(argv)

    from src.data import load_catalog

    cfg = load_config()
    queries, parsed = load_queries(), load_parsed()
    runs, info = run_systems(queries, parsed, cfg, depth=cfg["eval"]["pool_depth"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESULTS_DIR / "runs.csv", index=False)
    info.to_csv(RESULTS_DIR / "run_info.csv", index=False)
    print(f"saved {len(runs)} run rows for {info.query_id.nunique()} queries x {info.system.nunique()} systems "
          f"(config {runs.config_hash.iat[0]})")

    n = cfg["generation"]["top_n"]
    pd.set_option("display.width", 160)
    print("\nAll 25 queries, structural (no labels): returned < N items, empty results, pool size, latency")
    print(info.groupby("system").agg(
        short_lt_n=("n_returned", lambda s: int((s < n).sum())), empty=("n_returned", lambda s: int((s == 0).sum())),
        short_lt_10=("n_returned", lambda s: int((s < 10).sum())), median_pool=("pool_len", "median"),
        median_search_ms=("search_ms", "median"), p95_search_ms=("search_ms", lambda s: s.quantile(0.95)),
    ).round(2).to_string())
    s4 = info[info.system == "S4"]
    print("\nS4 post-filter K used:", s4.k_used.value_counts().sort_index().to_dict())
    pools = info.pivot(index="query_id", columns="system", values="pool_len")[["S4", "S5"]]
    print("Queries where S4 pool < S5 pool:", int((pools.S4 < pools.S5).sum()), "of", len(pools))

    split_q = queries[queries.split == args.split]
    adh = adherence_table(runs[runs.query_id.isin(split_q.query_id)], split_q, load_catalog())
    print(f"\n{args.split}: constraint adherence (mean over queries with gold limits that returned items)")
    print(adh.groupby("system")[["adherence@5", "adherence@10", "excl_violations@10"]].mean().round(3).to_string())

    gold = split_q
    versions = {"v2": parsed}
    v1_path = RESULTS_DIR / f"parsed_queries_{args.split}_v1.jsonl"
    if v1_path.exists():
        versions = {"v1": pd.read_json(v1_path, lines=True), **versions}
    print(f"\n{args.split}: parse micro-F1 (hard fields = price, category, brand)")
    for name, pred in versions.items():
        counts = parse_counts(pred[pred.query_id.isin(gold.query_id)], gold)
        hard = micro_f1(counts[counts.field.isin(HARD_FIELDS)])
        print(f"  {name}: hard F1 {hard['f1']:.3f} (P {hard['precision']:.3f}, R {hard['recall']:.3f}) | "
              f"with exclusions F1 {micro_f1(counts)['f1']:.3f} | per field: "
              + ", ".join(f"{r.field} {r.tp}/{r.fp}/{r.fn}" for r in counts.itertuples()))


if __name__ == "__main__":
    main()
