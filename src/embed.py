"""Encode products and queries with BGE; save and load the embedding matrix (spec section 6).

Run `python -m src.embed` to (re)build artifacts/embeddings.npy and artifacts/product_ids.npy.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src import ROOT, load_config

ARTIFACTS = ROOT / "artifacts"
# BGE's recommended retrieval instruction for short queries (spec 6). Products get no instruction.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=2)
def load_model(name: str = "BAAI/bge-small-en-v1.5") -> Any:
    """Load (once per process) a sentence-transformers model on CPU or GPU."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def encode_products(texts: list[str], model: Any, batch_size: int = 64,
                    show_progress_bar: bool = False) -> np.ndarray:
    """Encode product texts to an L2-normalized float32 matrix [N, dim]."""
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=show_progress_bar)
    return np.asarray(emb, dtype=np.float32)


def use_query_instruction(cfg: dict) -> bool:
    """Read embedding.query_instruction: true/false, or 'auto' = use the prefix until the
    dev-query comparison (spec 6) sets an explicit value in config.yaml."""
    setting = cfg["embedding"].get("query_instruction", "auto")
    if setting == "auto":
        return True
    if isinstance(setting, bool):
        return setting
    raise ValueError(f"embedding.query_instruction must be true, false or 'auto', got {setting!r}")


def encode_query(text: str, model: Any, use_instruction: bool = True) -> np.ndarray:
    """Encode one query to a normalized float32 vector [dim], optionally with the BGE prefix."""
    emb = model.encode([text], prompt=QUERY_INSTRUCTION if use_instruction else None,
                       normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(emb[0], dtype=np.float32)


def save_embeddings(embeddings: np.ndarray, product_ids: list[str], out_dir: Path = ARTIFACTS) -> None:
    """Save the matrix and its matching product IDs (row i <-> product_ids[i])."""
    if len(embeddings) != len(product_ids):
        raise ValueError("embeddings and product_ids must have the same length")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    np.save(out_dir / "product_ids.npy", np.asarray(product_ids, dtype=str))


def load_embeddings(out_dir: Path = ARTIFACTS) -> tuple[np.ndarray, np.ndarray]:
    """Load the matrix and product IDs saved by `save_embeddings`."""
    return np.load(out_dir / "embeddings.npy"), np.load(out_dir / "product_ids.npy")


def build(cfg: dict | None = None) -> np.ndarray:
    """Embed the processed catalog's product_text and persist the result."""
    from src.data import load_catalog

    cfg = cfg or load_config()
    catalog = load_catalog()
    model = load_model(cfg["embedding"]["model"])
    emb = encode_products(catalog["product_text"].tolist(), model,
                          batch_size=cfg["embedding"]["batch_size"], show_progress_bar=True)
    save_embeddings(emb, catalog["product_id"].tolist())
    return emb


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    matrix = build()
    print(f"embeddings {matrix.shape} {matrix.dtype} in {time.perf_counter() - start:.0f}s")
