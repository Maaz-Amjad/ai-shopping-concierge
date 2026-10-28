"""User preference vectors, personalized ranking (tech spec 8.4) and experiment B (spec 10.6).

final(p) = w_sem * S_sem + w_pers * S_pers + w_pop * S_pop, where S_sem and S_pers are min-max
normalized across the candidate pool and S_pop = log(1 + count) / log(1 + max_count).
Cold-start users get w_pers = 0 with the other weights renormalized. Ties: popularity, then product_id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluate import bootstrap_ci

MS_PER_DAY = 86_400_000


@dataclass(frozen=True)
class Weights:
    sem: float = 0.7
    pers: float = 0.25
    pop: float = 0.05

    @classmethod
    def from_config(cls, cfg: dict) -> Weights:
        r = cfg["ranking"]
        return cls(r["w_sem"], r["w_pers"], r["w_pop"])

    def cold_start(self) -> Weights:
        """Spec 8.4: w_pers = 0, remaining weights renormalized to sum to 1."""
        total = self.sem + self.pop
        return Weights(self.sem / total, 0.0, self.pop / total)


def minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; constant (or empty) input gives zeros (spec 8.4)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def popularity_score(counts: np.ndarray, max_count: float) -> np.ndarray:
    """S_pop = log(1 + count) / log(1 + max_count)."""
    return np.log1p(np.asarray(counts, dtype=np.float64)) / np.log1p(max_count)


def user_vector(rows: np.ndarray, embeddings: np.ndarray, timestamps: np.ndarray | None = None,
                half_life_days: float | None = None) -> np.ndarray | None:
    """Normalized (optionally recency-weighted) mean of the user's item embeddings; None if no history."""
    rows = np.asarray(rows, dtype=int)
    if len(rows) == 0:
        return None
    weights = np.ones(len(rows))
    if half_life_days and timestamps is not None:
        age_days = (np.max(timestamps) - np.asarray(timestamps)) / MS_PER_DAY
        weights = 0.5 ** (age_days / half_life_days)
    v = (weights[:, None] * embeddings[rows]).sum(axis=0)
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32) if norm > 0 else None


def build_profiles(interactions: pd.DataFrame, catalog: pd.DataFrame, embeddings: np.ndarray,
                   half_life_days: float | None = None) -> dict[str, np.ndarray]:
    """One preference vector per user from their interactions on catalog products."""
    row_of = pd.Series(np.arange(len(catalog)), index=catalog["product_id"].to_numpy())
    df = interactions[interactions["product_id"].isin(row_of.index)]
    profiles = {}
    for user, g in df.groupby("user_id", sort=True):
        v = user_vector(row_of.loc[g["product_id"]].to_numpy(), embeddings, g["timestamp"].to_numpy(),
                        half_life_days)
        if v is not None:
            profiles[user] = v
    return profiles


def rank(cands: pd.DataFrame, user_vec: np.ndarray | None, w: Weights, *, catalog: pd.DataFrame,
         embeddings: np.ndarray) -> pd.DataFrame:
    """Re-rank a candidate pool (columns row, score = cosine to the query) with score components (NFR-7)."""
    rows = cands["row"].to_numpy(dtype=int)
    w = w if user_vec is not None else w.cold_start()
    popularity = catalog["popularity"].to_numpy()
    out = pd.DataFrame({
        "row": rows,
        "product_id": catalog["product_id"].to_numpy()[rows],
        "cosine": cands["score"].to_numpy(dtype=np.float64),
        "s_sem": minmax(cands["score"].to_numpy()),
        "s_pers": minmax(embeddings[rows] @ user_vec) if user_vec is not None else np.zeros(len(rows)),
        "s_pop": popularity_score(popularity[rows], popularity.max()),
        "popularity": popularity[rows],
    })
    out["final"] = w.sem * out["s_sem"] + w.pers * out["s_pers"] + w.pop * out["s_pop"]
    out = out.sort_values(["final", "popularity", "product_id"], ascending=[False, False, True],
                          kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# ---- experiment B: held-out next item (spec 5.4, 10.6) ------------------------------------------

METHODS = ("random", "popularity", "profile", "blend")


def holdout_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each user's last interaction by time (ties: product_id) is held out; the rest is history."""
    df = interactions.sort_values(["user_id", "timestamp", "product_id"], kind="mergesort")
    last = df.groupby("user_id").tail(1).index
    return df.drop(index=last), df.loc[last]


def item_rank(scores: np.ndarray, target: int, popularity: np.ndarray, id_order: np.ndarray,
              candidates: np.ndarray) -> int:
    """1-based rank of `target` among `candidates` ordered by score desc, popularity desc, product_id asc."""
    s, p, o = scores[candidates], popularity[candidates], id_order[candidates]
    st, pt, ot = scores[target], popularity[target], id_order[target]
    better = (s > st) | ((s == st) & ((p > pt) | ((p == pt) & (o < ot))))
    return int(better.sum()) + 1


def next_item_eval(interactions: pd.DataFrame, catalog: pd.DataFrame, embeddings: np.ndarray, w: Weights,
                   seed: int = 42, k: int = 10, half_life_days: float | None = None) -> pd.DataFrame:
    """Rank all items the user has not seen for each user; one row per (user, method) with hit@k and ndcg@k.

    blend = w_pers * S_pers + w_pop * S_pop (no query, so no semantic term), renormalized; S_pers is
    min-max normalized over the user's unseen items.
    """
    rng = np.random.default_rng(seed)
    history, held = holdout_split(interactions)
    row_of = pd.Series(np.arange(len(catalog)), index=catalog["product_id"].to_numpy())
    popularity = catalog["popularity"].to_numpy()
    s_pop = popularity_score(popularity, popularity.max())
    id_order = np.argsort(np.argsort(catalog["product_id"].to_numpy()))
    total = w.pers + w.pop
    hist_by_user = {u: g for u, g in history.groupby("user_id", sort=True)}
    rows = []
    for _, h in held.sort_values("user_id").iterrows():
        g = hist_by_user.get(h["user_id"])
        if g is None:
            continue
        seen = row_of.loc[g["product_id"]].to_numpy()
        u = user_vector(seen, embeddings, g["timestamp"].to_numpy(), half_life_days)
        target = int(row_of.loc[h["product_id"]])
        if u is None or target in set(seen):
            continue
        unseen = np.setdiff1d(np.arange(len(catalog)), seen)
        cos = embeddings @ u
        pers = np.zeros(len(catalog))
        pers[unseen] = minmax(cos[unseen])
        scores = {"random": rng.random(len(catalog)), "popularity": s_pop, "profile": cos.astype(np.float64),
                  "blend": (w.pers * pers + w.pop * s_pop) / total}
        for method in METHODS:
            r = item_rank(scores[method], target, popularity, id_order, unseen)
            rows.append({"user_id": h["user_id"], "method": method, "rank": r, f"hit@{k}": float(r <= k),
                         f"ndcg@{k}": 1 / np.log2(r + 1) if r <= k else 0.0})
    return pd.DataFrame(rows)


def summarize_next_item(per_user: pd.DataFrame, k: int = 10, n_resamples: int = 1000,
                        seed: int = 42) -> pd.DataFrame:
    """Mean and 95% bootstrap CI (over users) of hit@k and ndcg@k per method."""
    out = []
    for method in METHODS:
        g = per_user[per_user["method"] == method]
        row = {"method": method, "users": len(g)}
        for m in (f"hit@{k}", f"ndcg@{k}"):
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = bootstrap_ci(g[m].to_numpy(), n_resamples, seed)
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    """python -m src.personalize: run experiment B on all users and save per-user and summary tables."""
    from src import ROOT, load_config, set_seed
    from src.data import load_catalog, load_interactions
    from src.embed import load_embeddings

    cfg = load_config()
    set_seed(cfg["seed"])
    catalog, interactions = load_catalog(), load_interactions()
    embeddings, product_ids = load_embeddings()
    if not (product_ids == catalog["product_id"].to_numpy()).all():
        raise ValueError("embeddings are not aligned with the catalog; rebuild with python -m src.embed")
    per_user = next_item_eval(interactions, catalog, embeddings, Weights.from_config(cfg), seed=cfg["seed"])
    summary = summarize_next_item(per_user, n_resamples=cfg["eval"]["bootstrap_resamples"], seed=cfg["seed"])
    out = ROOT / "eval" / "results"
    per_user.to_csv(out / "exp_b_per_user.csv", index=False)
    summary.to_csv(out / "exp_b_summary.csv", index=False)
    print(f"experiment B: {summary['users'].iat[0]} users, {len(catalog)} items, last interaction held out")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
