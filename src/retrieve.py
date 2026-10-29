"""Retrievers: TF-IDF (S1), BM25 (S2), dense unfiltered (S3), post-filter (S4), filter-first (S5).

All retrievers return a DataFrame with columns: row (catalog row index), score, rank (1-based).
Ties are broken by catalog row order, which is popularity desc then product_id (see src.data).
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.filters import build_mask, has_price_filter, unpriced_mask
from src.parse import ParsedQuery

MODES = ("dense", "post_filter", "filter_first")  # S3, S4, S5

# Keeps part numbers intact: "w10425238", "da61-06796a", "4.6".
TOKEN_PATTERN = r"[a-z0-9]+(?:[-.][a-z0-9]+)*"
_TOKEN_RE = re.compile(TOKEN_PATTERN)


def tokenize(text: str) -> list[str]:
    """Lowercase word/part-number tokens, shared by TF-IDF and BM25."""
    return _TOKEN_RE.findall(text.lower())


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min() if len(x) else 0.0
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def hybrid_scores(dense: np.ndarray, lexical: np.ndarray, alpha: float) -> np.ndarray:
    """Phase 13.5: (1 - alpha) * minmax(dense cosine) + alpha * minmax(BM25), over the same products."""
    return (1 - alpha) * _minmax(dense) + alpha * _minmax(lexical)


def top_k(scores: np.ndarray, k: int, rows: np.ndarray | None = None) -> pd.DataFrame:
    """Rank `scores` (aligned with `rows`, default 0..n-1) descending; ties by row index."""
    rows = np.arange(len(scores)) if rows is None else rows
    order = np.lexsort((rows, -scores))[:k]
    return pd.DataFrame({"row": rows[order], "score": scores[order], "rank": np.arange(1, len(order) + 1)})


class TfidfRetriever:
    """S1: TF-IDF over product_text, cosine similarity with the raw query."""

    def __init__(self, texts: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                          sublinear_tf=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform(texts)  # rows are L2-normalized

    def search(self, query: str, k: int) -> pd.DataFrame:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        return top_k(scores, k)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "TfidfRetriever":
        with open(path, "rb") as f:
            return pickle.load(f)


class BM25Retriever:
    """S2: Okapi BM25 over product_text with the raw query."""

    def __init__(self, texts: list[str]) -> None:
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> pd.DataFrame:
        return top_k(np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float64), k)


@dataclass
class DenseIndex:
    """Normalized embedding matrix plus a FAISS inner-product index (cosine similarity)."""

    embeddings: np.ndarray
    index: faiss.IndexFlatIP = field(init=False)

    def __post_init__(self) -> None:
        self.embeddings = np.ascontiguousarray(self.embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query_vec: np.ndarray, k: int) -> pd.DataFrame:
        """S3: unfiltered exact search with FAISS."""
        scores, rows = self.index.search(query_vec.reshape(1, -1).astype(np.float32), k)
        valid = rows[0] >= 0
        # Re-rank FAISS hits with the shared tie-break so all systems order ties identically.
        return top_k(scores[0][valid].astype(np.float64), k, rows[0][valid])

    def filter_first(self, query_vec: np.ndarray, mask: np.ndarray, k: int) -> pd.DataFrame:
        """S5: score only products that satisfy the hard constraints (spec 6)."""
        rows = np.flatnonzero(mask)
        scores = (self.embeddings[rows] @ query_vec).astype(np.float64)
        return top_k(scores, k, rows)

    def hybrid_filter_first(self, query_vec: np.ndarray, lexical_scores: np.ndarray, mask: np.ndarray, k: int,
                            alpha: float) -> pd.DataFrame:
        """Phase 13.5: filter-first with dense and keyword (BM25) scores blended over the allowed products only."""
        rows = np.flatnonzero(mask)
        dense = (self.embeddings[rows] @ query_vec).astype(np.float64)
        return top_k(hybrid_scores(dense, np.asarray(lexical_scores, dtype=np.float64)[rows], alpha), k, rows)

    def post_filter(self, query_vec: np.ndarray, mask: np.ndarray, n: int,
                    k0: int = 100, k_max: int = 1000) -> tuple[pd.DataFrame, int]:
        """S4 (proposal design): retrieve K, keep rows passing `mask`; double K until >= n survive or K hits k_max.

        Returns all survivors within the final K (the candidate pool) and the K that was used.
        """
        k = k0
        while True:
            hits = self.search(query_vec, min(k, len(self.embeddings)))
            kept = hits[mask[hits["row"].to_numpy()]].reset_index(drop=True)
            if len(kept) >= n or k >= k_max:
                kept["rank"] = np.arange(1, len(kept) + 1)
                return kept, min(k, k_max)
            k = min(k * 2, k_max)


@dataclass
class RetrievalResult:
    """Candidates plus the facts the evaluation and the answer need."""

    candidates: pd.DataFrame            # row, score, rank (best first)
    n_satisfying: int                   # products passing every hard constraint
    price_excluded: int                 # products dropped only because their price is missing
    unpriced: pd.DataFrame              # "price not listed" fallback list (never ranked or scored)
    k_used: int | None = None           # post-filter only: the K finally searched


def retrieve(pq: ParsedQuery, mode: str, pool_size: int, *, catalog: pd.DataFrame, index: DenseIndex,
             query_vec: np.ndarray, n: int = 5, k0: int = 100, k_max: int = 1000,
             use_exclusions: bool = True, lexical_scores: np.ndarray | None = None,
             alpha: float = 0.0) -> RetrievalResult:
    """Retrieve candidates for a parsed query (spec 11).

    mode: "dense" (S3, filters ignored), "post_filter" (S4), "filter_first" (S5).
    `query_vec` must encode the text that mode uses (raw query for S3, pq.semantic_query otherwise).
    The unpriced fallback list is filled only when a price filter leaves fewer than `n` candidates.
    filter_first with alpha > 0 and BM25 `lexical_scores` (one per catalog row) blends keyword matching in (13.5).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    mask = build_mask(catalog, pq, use_exclusions)
    no_price_filter = build_mask(catalog, pq.model_copy(update={"max_price": None, "min_price": None}),
                                 use_exclusions)
    price_excluded = int((no_price_filter & catalog["price"].isna().to_numpy()).sum()) \
        if has_price_filter(pq) else 0
    k_used = None
    if mode == "dense":
        cands = index.search(query_vec, pool_size)
    elif mode == "post_filter":
        cands, k_used = index.post_filter(query_vec, mask, n=n, k0=k0, k_max=k_max)
        cands = cands.head(pool_size)
    elif alpha > 0 and lexical_scores is not None:
        cands = index.hybrid_filter_first(query_vec, lexical_scores, mask, pool_size, alpha)
    else:
        cands = index.filter_first(query_vec, mask, pool_size)
    unpriced = pd.DataFrame(columns=["row", "score", "rank"])
    if mode != "dense" and has_price_filter(pq) and len(cands) < n:
        unpriced = index.filter_first(query_vec, unpriced_mask(catalog, pq, use_exclusions), n)
    return RetrievalResult(cands, int(mask.sum()), price_excluded, unpriced, k_used)
