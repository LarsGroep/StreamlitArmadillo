"""Shared fixtures: small synthetic cohorts so tests don't need the real CSV
(except the loader integration tests, which use it directly)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from armadillo_scoring.schema import ArtistRecord  # noqa: E402


@pytest.fixture
def full_record() -> ArtistRecord:
    """A record providing every canonical attribute."""
    return ArtistRecord(
        artist_id="full",
        attributes={
            "rank_trend": 0.9,
            "new_entry_rate": 0.8,
            "chart_quality": 0.7,
            "content_appeal": 0.6,
            "live_affinity": 0.5,
            "reach": 0.4,
            "catalog_breadth": 0.3,
            "recency": 0.2,
            "recent_activity": 0.1,
        },
        name="Full Coverage",
        source="test",
    )


@pytest.fixture
def sparse_record() -> ArtistRecord:
    """A record like a one-hit billboard artist: no trend, no audio."""
    return ArtistRecord(
        artist_id="sparse",
        attributes={
            "chart_quality": 0.55,
            "reach": 0.1,
            "catalog_breadth": 0.1,
            "new_entry_rate": 0.9,
            "recency": 0.4,
            "recent_activity": 0.0,
        },
    )


@pytest.fixture
def cohort(full_record, sparse_record) -> list[ArtistRecord]:
    """A small mixed cohort: full, sparse, and a mid-coverage artist."""
    mid = ArtistRecord(
        artist_id="mid",
        attributes={
            "rank_trend": 0.5,
            "new_entry_rate": 0.5,
            "chart_quality": 0.5,
            "reach": 0.5,
            "catalog_breadth": 0.5,
            "recency": 0.5,
            "recent_activity": 0.5,
        },
    )
    return [full_record, sparse_record, mid]
