"""armadillo_scoring.score — fold the composite signals into ONE speed score.

Input : the signal matrix from signals.py (index = artist_id, ~5 columns,
        values in [0, 1]).
Output: a DataFrame with, per artist:
        speed_score            weighted sum of the signals, in [0, 1]
        contrib_<signal>       weight x value for each signal — the entire
                               explanation. The contributions ADD UP to the
                               score exactly, so "why did artist X score 0.71?"
                               is answered by reading one row. No SHAP, no
                               model: the breakdown IS the arithmetic.

CAVEAT for PCA frames: a principal component mixes attributes with both signs,
so a contrib_pc* value is a valid share of the score but is NOT readable as
"more of attribute X = better" — that per-attribute story only holds for
manual-mode signals.

Weights resolution order:
  1. explicit `weights=` argument                      (caller knows best)
  2. schema.SIGNAL_WEIGHTS when columns match SIGNALS  (manual mode default)
  3. explained-variance ratios stored by PCA mode      (pca mode default)
  4. equal weights                                     (last resort)
"""

from __future__ import annotations

import math
import warnings
from typing import Mapping

import pandas as pd

from armadillo_scoring.schema import SIGNAL_WEIGHTS, SIGNALS

CONTRIB_PREFIX = "contrib_"
SCORE_COL = "speed_score"


def resolve_weights(
    signals: pd.DataFrame, weights: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Pick the weight vector for a signal matrix (see module docstring)."""
    columns = list(signals.columns)

    if weights is not None:
        missing = set(columns) - set(weights)
        if missing:
            raise ValueError(f"weights is missing entries for signals: {sorted(missing)}")
        chosen = {c: float(weights[c]) for c in columns}
    elif set(columns) == set(SIGNALS):
        chosen = {c: SIGNAL_WEIGHTS[c] for c in columns}
    elif "explained_variance_ratio" in signals.attrs:
        ratios = signals.attrs["explained_variance_ratio"]
        if len(ratios) != len(columns):
            raise ValueError("explained_variance_ratio does not match signal columns.")
        chosen = dict(zip(columns, (float(r) for r in ratios)))
    else:
        warnings.warn(
            f"Signal columns {sorted(columns)} match neither SIGNALS nor a PCA "
            "frame; falling back to equal weights. Pass weights= to control this.",
            stacklevel=2,
        )
        chosen = {c: 1.0 for c in columns}

    if any(w < 0 for w in chosen.values()):
        raise ValueError("Signal weights must be non-negative.")
    total = math.fsum(chosen.values())
    if total <= 0:
        raise ValueError("Signal weights must sum to a positive number.")
    return {c: w / total for c, w in chosen.items()}  # normalize to sum 1


def speed_score(
    signals: pd.DataFrame, weights: Mapping[str, float] | None = None
) -> pd.DataFrame:
    """Score every artist; deterministic, sorted best-first (ties by artist_id).

    Returns columns: speed_score, then contrib_<signal> per signal column.
    speed_score == sum of the contrib_ columns, exactly.
    """
    if signals.empty:
        raise ValueError("Cannot score an empty signal matrix.")
    if signals.isna().any().any():
        raise ValueError(
            "Signal matrix contains NaN. Run signals.to_signals(fill_missing=True) "
            "or fill them yourself before scoring."
        )

    resolved = resolve_weights(signals, weights)

    contribs = pd.DataFrame(
        {f"{CONTRIB_PREFIX}{c}": signals[c] * w for c, w in resolved.items()},
        index=signals.index,
    )
    out = contribs.copy()
    out.insert(0, SCORE_COL, contribs.sum(axis=1))
    out.attrs["weights"] = resolved

    # Deterministic order: best score first, artist_id breaks ties
    # (stable sort over an id-sorted frame).
    out = out.sort_index(kind="mergesort")
    return out.sort_values(SCORE_COL, ascending=False, kind="mergesort")


def explain(scored: pd.DataFrame, artist_id: str) -> str:
    """One artist's breakdown as a human-readable line block."""
    if artist_id not in scored.index:
        raise KeyError(f"artist_id '{artist_id}' not in scored frame.")
    row = scored.loc[artist_id]
    weights = scored.attrs.get("weights", {})
    lines = [f"{artist_id}: speed_score = {row[SCORE_COL]:.3f}"]
    contrib_cols = [c for c in scored.columns if c.startswith(CONTRIB_PREFIX)]
    for col in sorted(contrib_cols, key=lambda c: -row[c]):
        signal = col[len(CONTRIB_PREFIX):]
        weight = weights.get(signal)
        value = row[col] / weight if weight else float("nan")
        weight_txt = f"{weight:.2f}" if weight is not None else "?"
        lines.append(
            f"  {signal:<12} {row[col]:.3f}  (= weight {weight_txt} x value {value:.2f})"
        )
    return "\n".join(lines)
