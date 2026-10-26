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

# Keeps part numbers intact: "w10425238", "da61-06796a", "4.6".
TOKEN_PATTERN = r"[a-z0-9]+(?:[-.][a-z0-9]+)*"
_TOKEN_RE = re.compile(TOKEN_PATTERN)


def tokenize(text: str) -> list[str]:
    """Lowercase word/part-number tokens, shared by TF-IDF and BM25."""
    return _TOKEN_RE.findall(text.lower())


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

    def post_filter(self, query_vec: np.ndarray, mask: np.ndarray, n: int,
                    k0: int = 100, k_max: int = 1000) -> tuple[pd.DataFrame, int]:
        """S4 (proposal design): retrieve K, keep rows passing `mask`; double K until >= n survive or K hits k_max.

        Returns the surviving top-n and the K that was finally used.
        """
        k = k0
        while True:
            hits = self.search(query_vec, min(k, len(self.embeddings)))
            kept = hits[mask[hits["row"].to_numpy()]].head(n).reset_index(drop=True)
            if len(kept) >= n or k >= k_max:
                kept["rank"] = np.arange(1, len(kept) + 1)
                return kept, min(k, k_max)
            k = min(k * 2, k_max)
