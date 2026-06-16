"""Billboard loader — the ONLY billboard-specific file in the kit.

Maps archive/billboard_24years_lyrics_spotify.csv (one row per song-on-chart,
years 2000-2023) onto the canonical ArtistRecord schema. Everything downstream
(signals, score, validate) never sees a billboard column name.

WHAT THE RAW DATA LOOKS LIKE
    ranking      1..100 chart position, 1 = best  (always present)
    band_singer  artist name                       (always present)
    song, year   track title, chart year           (always present)
    danceability..duration_ms  Spotify audio features — present in only ~14%
                 of rows, almost all 2000-2004. Treated as optional enrichment.

MAPPING (billboard column -> canonical attribute)
    growth.rank_trend       slope of inverted rank over the artist's years
    growth.new_entry_rate   chart entries per active year
    engagement.chart_quality   mean inverted rank of all entries
    engagement.content_appeal  mean of danceability/energy/valence (if audio)
    live.live_affinity      mean audio 'liveness' (if audio)
    audience.reach          number of chart entries (percentile-ranked)
    audience.catalog_breadth   distinct songs + distinct chart years (pct-ranked)
    momentum.recency        last active year, linearly mapped onto [0, 1]
    momentum.recent_activity   entries within the trailing window (pct-ranked)

NORMALIZATION POLICY
    Everything is emitted in [0, 1], oriented "higher = more promising"
    (schema contract). Rank 1..100 is inverted via (101 - rank) / 100.
    Unbounded counts are percentile-ranked WITHIN the loaded population —
    that makes scores relative to the cohort, which is exactly what a talent
    scout comparing artists wants. Attributes that cannot be computed for an
    artist (single entry -> no trend; no audio rows -> no appeal/liveness)
    are OMITTED, never invented.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from armadillo_scoring.schema import ArtistRecord

# Raw columns we consume. Anything else (urls, lyrics, ids...) is ignored.
ARTIST_COL = "band_singer"
RANK_COL = "ranking"
SONG_COL = "song"
YEAR_COL = "year"
AUDIO_APPEAL_COLS = ("danceability", "energy", "valence")
AUDIO_LIVENESS_COL = "liveness"

RANK_WORST = 100  # billboard year-end Hot 100

# "Recent" for momentum.recent_activity = this many trailing years of the data.
RECENT_WINDOW_YEARS = 3

DEFAULT_CSV = (
    Path(__file__).resolve().parents[2] / "archive" / "billboard_24years_lyrics_spotify.csv"
)


def _invert_rank(rank: pd.Series) -> pd.Series:
    """Map rank 1..100 -> (0, 1], oriented higher = better (rank 1 -> 1.0)."""
    return (RANK_WORST + 1 - rank) / RANK_WORST


def _pct_rank(values: pd.Series) -> pd.Series:
    """Percentile-rank a count-like series onto [0, 1] within the cohort."""
    return values.rank(pct=True, method="average")


def _read_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read the CSV without treating artists literally named 'NA' / 'None' /
    'null' as missing values; every other column keeps default NA parsing."""
    try:
        from pandas.io.parsers.readers import STR_NA_VALUES
    except ImportError:  # private-ish path; fall back to plain parsing
        return pd.read_csv(csv_path)
    columns = pd.read_csv(csv_path, nrows=0).columns
    na_values = {c: list(STR_NA_VALUES) for c in columns if c != ARTIST_COL}
    na_values[ARTIST_COL] = [""]  # a truly empty artist cell is still missing
    return pd.read_csv(csv_path, keep_default_na=False, na_values=na_values)


def _rank_trend_slope(years: np.ndarray, inv_ranks: np.ndarray) -> float | None:
    """OLS slope of inverted rank vs. year; None when a trend is undefined.

    Needs at least two DISTINCT years (two songs in the same year carry no
    trajectory information).
    """
    if len(years) < 2 or np.unique(years).size < 2:
        return None
    slope = np.polyfit(years.astype(float), inv_ranks.astype(float), 1)[0]
    return float(slope)


def load(csv_path: str | Path = DEFAULT_CSV) -> list[ArtistRecord]:
    """Load the billboard CSV and return one canonical ArtistRecord per artist."""
    df = _read_csv(csv_path)

    required = {ARTIST_COL, RANK_COL, SONG_COL, YEAR_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Billboard CSV is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=[ARTIST_COL, RANK_COL, YEAR_COL]).copy()
    df["_inv_rank"] = _invert_rank(df[RANK_COL])

    max_year = int(df[YEAR_COL].max())
    min_year = int(df[YEAR_COL].min())
    year_span = max(max_year - min_year, 1)
    recent_cutoff = max_year - RECENT_WINDOW_YEARS + 1

    grouped = df.groupby(ARTIST_COL, sort=True)

    # --- per-artist scalars, vectorized where possible -----------------------
    n_entries = grouped.size()
    n_songs = grouped[SONG_COL].nunique()
    n_years = grouped[YEAR_COL].nunique()
    last_year = grouped[YEAR_COL].max()
    first_year = grouped[YEAR_COL].min()
    mean_inv_rank = grouped["_inv_rank"].mean()
    best_rank = grouped[RANK_COL].min()
    recent_entries = (
        df[YEAR_COL].ge(recent_cutoff).groupby(df[ARTIST_COL]).sum()
    )

    # entries per active year (active span is inclusive: 2000..2002 = 3 years)
    active_span = (last_year - first_year + 1).astype(float)
    entry_rate = n_entries / active_span

    # cohort-relative percentile ranks for the unbounded counts
    reach = _pct_rank(n_entries)
    catalog_breadth = _pct_rank(n_songs + n_years)
    new_entry_rate = _pct_rank(entry_rate)
    recency = (last_year - min_year) / year_span
    # Artists with zero recent entries share a fixed 0.0 rather than a
    # percentile, so "inactive" reads as inactive regardless of cohort shape.
    recent_activity = _pct_rank(recent_entries.where(recent_entries > 0)).fillna(0.0)

    # --- audio enrichment (sparse: ~14% of rows, mostly 2000-2004) -----------
    # Only ever computed from columns the CSV actually has ("omitted, never
    # invented"): a source with appeal columns but no liveness column emits
    # content_appeal and simply skips live_affinity.
    appeal_cols = [c for c in AUDIO_APPEAL_COLS if c in df.columns]
    has_liveness = AUDIO_LIVENESS_COL in df.columns
    audio_cols = appeal_cols + ([AUDIO_LIVENESS_COL] if has_liveness else [])
    has_audio = df[audio_cols].notna().all(axis=1) if audio_cols else pd.Series(False, index=df.index)
    audio_df = df[has_audio]
    content_appeal = pd.Series(dtype=float)
    live_affinity = pd.Series(dtype=float)
    if not audio_df.empty:
        audio_grouped = audio_df.groupby(ARTIST_COL)
        if appeal_cols:
            content_appeal = audio_grouped[appeal_cols].mean().mean(axis=1)
        if has_liveness:
            live_affinity = audio_grouped[AUDIO_LIVENESS_COL].mean()

    # --- rank trend needs per-artist regression ------------------------------
    rank_trend_raw: dict[str, float] = {}
    for artist, g in grouped:
        slope = _rank_trend_slope(g[YEAR_COL].to_numpy(), g["_inv_rank"].to_numpy())
        if slope is not None:
            rank_trend_raw[artist] = slope
    # Slopes are small signed numbers; percentile-rank them so the attribute
    # lands in [0, 1] with 0.5 ~ "flat trajectory within this cohort".
    rank_trend = _pct_rank(pd.Series(rank_trend_raw, dtype=float))

    # --- assemble records -----------------------------------------------------
    records: list[ArtistRecord] = []
    for artist in n_entries.index:
        attributes: dict[str, float] = {
            "chart_quality": float(mean_inv_rank[artist]),
            "reach": float(reach[artist]),
            "catalog_breadth": float(catalog_breadth[artist]),
            "new_entry_rate": float(new_entry_rate[artist]),
            "recency": float(recency[artist]),
            "recent_activity": float(recent_activity[artist]),
        }
        if artist in rank_trend.index:
            attributes["rank_trend"] = float(rank_trend[artist])
        if artist in content_appeal.index:
            attributes["content_appeal"] = float(np.clip(content_appeal[artist], 0.0, 1.0))
        if artist in live_affinity.index:
            attributes["live_affinity"] = float(np.clip(live_affinity[artist], 0.0, 1.0))

        records.append(
            ArtistRecord(
                artist_id=str(artist),
                attributes=attributes,
                name=str(artist),
                source="billboard",
                meta={
                    "entries": int(n_entries[artist]),
                    "first_year": int(first_year[artist]),
                    "last_year": int(last_year[artist]),
                    "best_rank": int(best_rank[artist]),
                },
            )
        )
    return records


def load_raw(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """The raw per-song table, for building proxy labels in validate.py."""
    return _read_csv(csv_path)
