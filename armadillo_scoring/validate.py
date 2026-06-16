"""armadillo_scoring.validate — does the speed score separate hits from non-hits?

This is the kit's yardstick, NOT a production-accuracy claim.

PROXY LABEL
    We have no ground-truth "this artist broke through" label, so we build a
    proxy from chart position: an artist is a HIT if their best (lowest)
    chart position ever reached a threshold (default: top 10). Alternatively,
    `top_fraction` labels the best X% of artists as hits.

METRICS
    auc            Probability that a random hit outscores a random non-hit
                   (Mann-Whitney formulation, ties counted half). 0.5 = the
                   score knows nothing, 1.0 = perfect separation.
    precision@k    Of the top-k artists by score, the share that are hits —
                   "if the scout meets the top k, how many are real?"

HONESTY CLAUSE (also in the README)
    Some signals are derived from chart data and the proxy label is TOO, so a
    high AUC here demonstrates the pipeline is wired correctly and produces a
    discriminative ranking — it does NOT prove the score predicts future
    breakout. That claim needs time-split validation on a source with real
    outcome data (e.g. Chartmetric history).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Proxy labels
# --------------------------------------------------------------------------- #
def hit_labels_from_chart(
    entries: pd.DataFrame,
    *,
    artist_col: str,
    rank_col: str,
    top_rank: int | None = 10,
    top_fraction: float | None = None,
) -> pd.Series:
    """Build the proxy label from any per-entry chart table (source-agnostic:
    it only needs an artist column and a rank column where LOWER = BETTER).

    top_rank      hit = artist's best rank <= top_rank   (default: top 10)
    top_fraction  hit = artist's best rank is at or below the top_fraction
                  quantile of best ranks (mutually exclusive with top_rank).
                  Best ranks are tied integers and ALL artists tied at the
                  cutoff are labeled hits, so the labeled share can exceed
                  top_fraction — e.g. a tiny fraction still labels every
                  rank-1 artist a hit. Deliberate: splitting tied artists
                  into different labels would be arbitrary.

    Returns Series: index = artist id, values 1 (hit) / 0 (non-hit).
    """
    if (top_rank is None) == (top_fraction is None):
        raise ValueError("Specify exactly one of top_rank or top_fraction.")

    best = entries.groupby(artist_col)[rank_col].min()

    if top_rank is not None:
        labels = (best <= top_rank).astype(int)
    else:
        if not 0.0 < top_fraction < 1.0:
            raise ValueError("top_fraction must be in (0, 1).")
        cutoff = best.quantile(top_fraction)  # lower rank = better
        labels = (best <= cutoff).astype(int)

    labels.index = labels.index.astype(str)
    labels.name = "is_hit"
    return labels


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def auc(scores: pd.Series, labels: pd.Series) -> float:
    """ROC-AUC via the rank (Mann-Whitney U) formulation. Pure numpy/pandas.

    scores and labels are aligned on their index; artists missing from either
    side are dropped.
    """
    aligned = pd.concat({"score": scores, "label": labels}, axis=1).dropna()
    y = aligned["label"].to_numpy()
    s = aligned["score"].to_numpy(dtype=float)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"AUC needs both classes (got {n_pos} hits, {n_neg} non-hits)."
        )

    ranks = pd.Series(s).rank(method="average").to_numpy()  # ties share ranks
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def precision_at_k(scores: pd.Series, labels: pd.Series, k: int) -> float:
    """Share of hits among the k highest-scored artists.

    Deterministic under score ties: ties are broken by artist id, exactly like
    score.speed_score orders its output.
    """
    aligned = pd.concat({"score": scores, "label": labels}, axis=1).dropna()
    if k <= 0:
        raise ValueError("k must be positive.")
    if aligned.empty:
        raise ValueError(
            "precision_at_k: scores and labels share no artists after index "
            "alignment — nothing to rank."
        )
    k = min(k, len(aligned))
    top = (
        aligned.sort_index(kind="mergesort")
        .sort_values("score", ascending=False, kind="mergesort")
        .head(k)
    )
    return float(top["label"].sum() / k)


@dataclass(frozen=True)
class ValidationReport:
    auc: float
    precision_at_k: dict[int, float]
    n_artists: int
    n_hits: int
    base_rate: float  # hits / artists — the "random scout" baseline

    def __str__(self) -> str:
        lines = [
            f"artists evaluated : {self.n_artists}",
            f"hits (proxy)      : {self.n_hits}  (base rate {self.base_rate:.1%})",
            f"AUC               : {self.auc:.3f}   (0.5 = random, 1.0 = perfect)",
        ]
        for k, p in self.precision_at_k.items():
            lift = p / self.base_rate if self.base_rate > 0 else float("nan")
            lines.append(f"precision@{k:<3}      : {p:.1%}  ({lift:.1f}x base rate)")
        return "\n".join(lines)


def evaluate(
    scores: pd.Series,
    labels: pd.Series | Mapping[str, int],
    ks: tuple[int, ...] = (10, 25, 50),
) -> ValidationReport:
    """Full report: AUC + precision@k for each k, plus base-rate context."""
    if not isinstance(labels, pd.Series):
        labels = pd.Series(dict(labels), name="is_hit")

    aligned = pd.concat({"score": scores, "label": labels}, axis=1).dropna()
    n = len(aligned)
    n_hits = int(aligned["label"].sum())

    return ValidationReport(
        auc=auc(aligned["score"], aligned["label"]),
        precision_at_k={k: precision_at_k(aligned["score"], aligned["label"], k) for k in ks},
        n_artists=n,
        n_hits=n_hits,
        base_rate=n_hits / n if n else float("nan"),
    )
