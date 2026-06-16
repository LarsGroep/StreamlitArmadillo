"""armadillo_scoring.schema — the canonical, source-agnostic input contract.

This module is the heart of the kit. It defines:

1. ArtistRecord       — the canonical unit the whole pipeline speaks:
                        an `artist_id` plus a dict of NORMALIZED numeric attributes.
2. CANONICAL_ATTRIBUTES — the controlled vocabulary of attribute names a loader
                        may emit, each tagged with the composite signal it feeds.
3. SIGNAL_CONFIG      — which attributes ("rules") roll up into which of the
                        five composite signals (growth, engagement, live,
                        audience, momentum).
4. SIGNAL_WEIGHTS     — how the five signals combine into one speed score.
5. validate_*()       — guards so a malformed source can't silently poison scores.

DESIGN CONTRACT (read before writing a new loader)
--------------------------------------------------
* A loader's ONLY job is to turn a raw source into a list of ArtistRecords whose
  attribute keys come from CANONICAL_ATTRIBUTES.
* Attributes are NORMALIZED and ORIENTED: every value is scaled to roughly
  [0, 1] and oriented so that HIGHER ALWAYS MEANS "more promising". A loader that
  reads a metric where low is good (e.g. a chart rank where 1 is best) must invert
  it before emitting. This keeps signals.py and score.py a plain weighted average —
  no per-attribute polarity bookkeeping downstream.
* A loader need NOT fill every attribute. Sparse / unavailable attributes
  (e.g. audio "live" features that the billboard CSV only has for 2000-2004) are
  simply omitted; signals.py averages over whatever is present.
* Source-agnostic means: a Chartmetric loader emits the SAME attribute names from
  its own raw metrics. The core never learns where the numbers came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Iterable, Mapping

# --------------------------------------------------------------------------- #
# 1. The five composite signals (the only signals the score ever sees).
# --------------------------------------------------------------------------- #
GROWTH = "growth"
ENGAGEMENT = "engagement"
LIVE = "live"
AUDIENCE = "audience"
MOMENTUM = "momentum"

SIGNALS: tuple[str, ...] = (GROWTH, ENGAGEMENT, LIVE, AUDIENCE, MOMENTUM)


# --------------------------------------------------------------------------- #
# 2. Canonical attribute vocabulary.
#    Each attribute is normalized to ~[0, 1] and oriented "higher = better".
#    `signal` says which composite it rolls up into.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Attribute:
    """One canonical, source-agnostic input feature."""

    name: str
    signal: str
    description: str  # what it means + the "higher = better" orientation


CANONICAL_ATTRIBUTES: dict[str, Attribute] = {
    attr.name: attr
    for attr in (
        # --- growth: is the artist's trajectory improving? ---
        Attribute(
            "rank_trend", GROWTH,
            "Trend of chart position over the artist's timeline. "
            "Higher = positions are getting better over time.",
        ),
        Attribute(
            "new_entry_rate", GROWTH,
            "Rate of fresh chart entries per active period. "
            "Higher = the artist keeps landing new things.",
        ),
        # --- engagement: how strongly does the content land? ---
        Attribute(
            "chart_quality", ENGAGEMENT,
            "Typical chart height of the artist's entries. "
            "Higher = entries chart higher on average.",
        ),
        Attribute(
            "content_appeal", ENGAGEMENT,
            "Intrinsic appeal of the content itself (e.g. an audio "
            "danceability/energy/valence blend). Higher = more appealing.",
        ),
        # --- live: live-performance dimension (often sparse on billboard) ---
        Attribute(
            "live_affinity", LIVE,
            "Live-performance signal of the content (e.g. audio 'liveness'). "
            "Higher = more of a live feel. Optional / source-dependent.",
        ),
        # --- audience: how broad is the footprint? ---
        Attribute(
            "reach", AUDIENCE,
            "Breadth of charting footprint (e.g. number of chart entries). "
            "Higher = larger reach.",
        ),
        Attribute(
            "catalog_breadth", AUDIENCE,
            "Distinct works and seasons the artist has charted. "
            "Higher = a broader, more sustained catalog.",
        ),
        # --- momentum: how hot is the artist right now? ---
        Attribute(
            "recency", MOMENTUM,
            "How recently the artist was active. "
            "Higher = more recent.",
        ),
        Attribute(
            "recent_activity", MOMENTUM,
            "Volume of activity inside the recent window. "
            "Higher = more currently active.",
        ),
    )
}


# --------------------------------------------------------------------------- #
# 3. SIGNAL_CONFIG — which attributes feed which composite signal.
#    Mode (a) of signals.py ("manual grouping") reads exactly this.
#    Derived from CANONICAL_ATTRIBUTES so the two can never drift apart.
# --------------------------------------------------------------------------- #
SIGNAL_CONFIG: dict[str, tuple[str, ...]] = {
    signal: tuple(a.name for a in CANONICAL_ATTRIBUTES.values() if a.signal == signal)
    for signal in SIGNALS
}


# --------------------------------------------------------------------------- #
# 4. SIGNAL_WEIGHTS — how the five signals combine into the single speed score.
#    "No 20 loose weights": the whole model is tuned by these FIVE numbers.
#    Must be non-negative and sum to 1.0 (see validate_signal_weights()).
# --------------------------------------------------------------------------- #
SIGNAL_WEIGHTS: dict[str, float] = {
    GROWTH: 0.25,
    MOMENTUM: 0.25,
    AUDIENCE: 0.20,
    ENGAGEMENT: 0.20,
    LIVE: 0.10,
}

# Loaders normalize to this range. Out-of-range values are NOT flagged by default
# (a z-score-style loader is allowed, it just forgoes the [0,1] guarantee); pass
# strict_range=True to validate_record()/validate_records() to report them.
NORMALIZED_RANGE: tuple[float, float] = (0.0, 1.0)


# --------------------------------------------------------------------------- #
# The canonical record.
# --------------------------------------------------------------------------- #
@dataclass
class ArtistRecord:
    """One artist, ready to score.

    artist_id   stable, source-unique identifier (string).
    attributes  normalized numeric features keyed by CANONICAL_ATTRIBUTES names.
                Missing keys are allowed (treated as "no signal" downstream).
    name        human-readable label (optional, never used by the math).
    source      provenance tag, e.g. "billboard" / "chartmetric" (optional).
    """

    artist_id: str
    attributes: dict[str, float]
    name: str | None = None
    source: str | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 5. Validation helpers.
# --------------------------------------------------------------------------- #
def validate_signal_weights(weights: Mapping[str, float] = SIGNAL_WEIGHTS) -> None:
    """Raise ValueError if the signal weights are not a clean probability split."""
    unknown = set(weights) - set(SIGNALS)
    if unknown:
        raise ValueError(f"Unknown signals in weights: {sorted(unknown)}")
    missing = set(SIGNALS) - set(weights)
    if missing:
        raise ValueError(f"Missing weights for signals: {sorted(missing)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Signal weights must be non-negative.")
    total = math.fsum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"Signal weights must sum to 1.0 (got {total!r}).")


def _is_finite_number(value) -> bool:
    # bool is a subclass of int; reject it so True/False can't masquerade as data.
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        # e.g. an int too large for float — report it, don't crash the batch.
        return False


def validate_record(record: ArtistRecord, *, strict_range: bool = False) -> list[str]:
    """Return a list of human-readable problems with one record (empty == ok).

    Errors (always reported): missing/blank id, non-mapping attributes,
    non-finite or non-numeric values, unknown attribute names.
    Range check (only when strict_range): values outside NORMALIZED_RANGE.
    """
    problems: list[str] = []

    if not isinstance(record.artist_id, str) or not record.artist_id.strip():
        problems.append("artist_id must be a non-empty string.")

    if not isinstance(record.attributes, Mapping):
        problems.append("attributes must be a mapping of name -> number.")
        return problems  # nothing else is checkable

    if not record.attributes:
        problems.append("attributes is empty: nothing to score.")

    lo, hi = NORMALIZED_RANGE
    for key, value in record.attributes.items():
        if key not in CANONICAL_ATTRIBUTES:
            problems.append(f"unknown attribute '{key}' (not in CANONICAL_ATTRIBUTES).")
            continue
        if not _is_finite_number(value):
            problems.append(f"attribute '{key}' must be a finite number, got {value!r}.")
            continue
        if strict_range and not (lo <= float(value) <= hi):
            problems.append(f"attribute '{key}'={value} outside normalized range {NORMALIZED_RANGE}.")

    return problems


def validate_records(
    records: Iterable[ArtistRecord], *, strict_range: bool = False
) -> dict[str, list[str]]:
    """Validate many records. Returns {artist_id: [problems]} for failing ones only.

    Also flags duplicate artist_ids, which would silently collapse downstream.
    """
    report: dict[str, list[str]] = {}
    seen: set[str] = set()
    for i, record in enumerate(records):
        problems = validate_record(record, strict_range=strict_range)
        key = record.artist_id if isinstance(record.artist_id, str) and record.artist_id else f"<row {i}>"
        if key in seen:
            problems.append(f"duplicate artist_id '{key}'.")
        seen.add(key)
        if problems:
            report.setdefault(key, []).extend(problems)
    return report


# --------------------------------------------------------------------------- #
# Small convenience accessors (used by signals.py / score.py).
# --------------------------------------------------------------------------- #
def known_attributes() -> tuple[str, ...]:
    """All canonical attribute names, in declaration order."""
    return tuple(CANONICAL_ATTRIBUTES)


def attributes_for_signal(signal: str) -> tuple[str, ...]:
    """The attribute names that roll up into a given composite signal."""
    if signal not in SIGNAL_CONFIG:
        raise KeyError(f"Unknown signal '{signal}'. Known: {SIGNALS}")
    return SIGNAL_CONFIG[signal]


def coverage(record: ArtistRecord) -> float:
    """Fraction of canonical attributes this record actually provides (0..1).

    Handy for diagnostics — e.g. billboard records will have low 'live' coverage.
    """
    if not isinstance(record.attributes, Mapping):
        return 0.0
    present = sum(1 for k in record.attributes if k in CANONICAL_ATTRIBUTES)
    return present / len(CANONICAL_ATTRIBUTES)


# Fail fast at import time if someone edits the weights into an invalid state.
validate_signal_weights()
