"""armadillo_scoring.signals — reduce canonical attributes to ~5 composite signals.

Input : list[ArtistRecord]   (any source, see schema.py)
Output: pandas DataFrame, index = artist_id, exactly one column per signal,
        every value in [0, 1].

Two reduction modes:

  mode="manual"  (a) Group attributes by SIGNAL_CONFIG and average within each
                 group. Transparent: a scout can read "growth = mean(rank_trend,
                 new_entry_rate)" straight off the config. This is the default.

  mode="pca"     (b) Principal components over the attribute matrix, keeping
                 N_SIGNALS components. Data-driven sanity check on the manual
                 grouping — if PCA scores rank artists wildly differently, the
                 manual config deserves a second look. Columns are named
                 pc1..pc5; their explained-variance ratios are stored in
                 result.attrs["explained_variance_ratio"] so score.py can
                 weight them accordingly.

MISSING DATA POLICY
    Loaders omit attributes they cannot compute (schema contract). Here:
    - manual mode averages over the attributes that ARE present per signal;
    - any signal/component still undefined is filled with the population
      MEDIAN of that column ("no information -> assume typical"), so every
      artist ends up with the same 5-signal shape and scores stay comparable.
    Pass fill_missing=False to keep NaNs and handle them yourself.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from armadillo_scoring.schema import (
    SIGNAL_CONFIG,
    SIGNALS,
    ArtistRecord,
    known_attributes,
)

N_SIGNALS = len(SIGNALS)

MODES = ("manual", "pca")


def attribute_frame(records: Iterable[ArtistRecord]) -> pd.DataFrame:
    """Stack records into a DataFrame: index=artist_id, one column per
    canonical attribute (NaN where a record omitted it)."""
    records = list(records)
    data = {rec.artist_id: rec.attributes for rec in records}
    frame = pd.DataFrame.from_dict(data, orient="index", dtype=float)
    # Stable, schema-defined column order; add all-NaN columns for attributes
    # no record provided so both modes see a fixed-width matrix. Reindexing on
    # the original ids too, because from_dict(orient="index") drops keys whose
    # inner dict is empty — those records must survive as all-NaN rows.
    return frame.reindex(index=list(data), columns=list(known_attributes()))


def to_signals(
    records: Iterable[ArtistRecord] | pd.DataFrame,
    mode: str = "manual",
    *,
    signal_config: Mapping[str, Sequence[str]] = SIGNAL_CONFIG,
    fill_missing: bool = True,
) -> pd.DataFrame:
    """Reduce canonical attributes to the composite signal matrix.

    Accepts either ArtistRecords or a pre-built attribute_frame().
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {MODES}.")

    attrs = records if isinstance(records, pd.DataFrame) else attribute_frame(records)
    if attrs.empty:
        raise ValueError("No records to reduce — got an empty input.")

    if mode == "manual":
        signals = _manual_signals(attrs, signal_config)
    else:
        signals = _pca_signals(attrs)

    if fill_missing:
        medians = signals.median()
        # A column that is ALL NaN has no median; neutral 0.5 is the only
        # honest stand-in ("we know nothing about this dimension").
        signals = signals.fillna(medians).fillna(0.5)
    return signals


# --------------------------------------------------------------------------- #
# (a) manual grouping
# --------------------------------------------------------------------------- #
def _manual_signals(
    attrs: pd.DataFrame, signal_config: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    unknown = {a for members in signal_config.values() for a in members} - set(attrs.columns)
    if unknown:
        raise ValueError(
            f"signal_config references attributes missing from the matrix: {sorted(unknown)}"
        )
    out = {}
    for signal, members in signal_config.items():
        if not members:
            raise ValueError(f"Signal '{signal}' has no attributes in the config.")
        # mean over the attributes this artist actually has; NaN if none.
        out[signal] = attrs[list(members)].mean(axis=1)
    return pd.DataFrame(out, index=attrs.index)


# --------------------------------------------------------------------------- #
# (b) PCA
# --------------------------------------------------------------------------- #
def _pca_signals(attrs: pd.DataFrame) -> pd.DataFrame:
    """PCA on the (median-imputed, standardized) attribute matrix.

    Deterministic: full SVD plus a contract-anchored sign convention — every
    attribute is oriented higher=better (schema contract), so each component
    is flipped to align non-negatively with the average attribute level. That
    matters because score.py weights every component POSITIVELY (by explained
    variance); an anti-oriented component would actively subtract signal.
    """
    # PCA cannot digest NaNs: impute column medians up front (all-NaN columns
    # carry no variance and are dropped from the decomposition).
    filled = attrs.fillna(attrs.median())
    filled = filled.dropna(axis=1, how="all")
    if filled.shape[1] == 0:
        raise ValueError("PCA mode needs at least one attribute with data.")

    n_components = min(N_SIGNALS, filled.shape[1], len(filled))

    values = filled.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0.0] = 1.0  # constant columns -> zero contribution, no div-by-0
    z = (values - mean) / std

    # Economy SVD of the centered/scaled matrix == PCA, deterministic.
    _, singular, vt = np.linalg.svd(z, full_matrices=False)

    # Rank cap: after centering, rank <= n_samples - 1; trailing ~zero singular
    # values are float noise that the min-max below would amplify to [0, 1].
    tol = singular[0] * 1e-10 if singular.size and singular[0] > 0 else 0.0
    n_components = min(n_components, max(int((singular > tol).sum()), 1))
    components = vt[:n_components]

    # Orient each component against the schema contract: attributes are
    # higher = better, so component scores should align non-negatively with
    # the row-mean of the standardized attributes. This resolves the SVD sign
    # ambiguity deterministically AND keeps positively-weighted components
    # from rewarding "less promising" directions.
    reference = z.mean(axis=1)
    for row in components:
        alignment = float((z @ row) @ reference)
        if alignment < 0:
            row *= -1.0
        elif alignment == 0:  # degenerate tie: fall back to a stable pivot
            pivot = np.argmax(np.abs(row))
            if row[pivot] < 0:
                row *= -1.0

    scores = z @ components.T

    # Min-max each component onto [0, 1] so PCA signals obey the same range
    # contract as manual ones. NOTE: a component mixes attributes with both
    # signs, so unlike manual signals a pc column is NOT per-attribute
    # monotone "higher = better" — treat PCA mode as a ranking cross-check,
    # not an attribute-level explanation.
    lo, hi = scores.min(axis=0), scores.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (scores - lo) / span

    total_var = (singular**2).sum()
    explained = (singular[:n_components] ** 2) / total_var if total_var > 0 else np.full(
        n_components, 1.0 / n_components
    )

    out = pd.DataFrame(
        scaled,
        index=attrs.index,
        columns=[f"pc{i + 1}" for i in range(n_components)],
    )
    out.attrs["explained_variance_ratio"] = [float(v) for v in explained]
    out.attrs["pca_loadings"] = pd.DataFrame(
        components, index=out.columns, columns=filled.columns
    )
    return out
