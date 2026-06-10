"""schema.py — contract validation."""

import math

import pytest

from armadillo_scoring import schema
from armadillo_scoring.schema import (
    ArtistRecord,
    SIGNAL_CONFIG,
    SIGNAL_WEIGHTS,
    SIGNALS,
    validate_record,
    validate_records,
    validate_signal_weights,
)


def test_signal_config_covers_all_signals_and_attributes():
    assert set(SIGNAL_CONFIG) == set(SIGNALS)
    grouped = {a for members in SIGNAL_CONFIG.values() for a in members}
    assert grouped == set(schema.known_attributes())
    # every signal has at least one attribute feeding it
    assert all(len(members) >= 1 for members in SIGNAL_CONFIG.values())


def test_default_weights_are_valid():
    validate_signal_weights()  # must not raise
    assert math.isclose(sum(SIGNAL_WEIGHTS.values()), 1.0)


@pytest.mark.parametrize(
    "weights",
    [
        {**SIGNAL_WEIGHTS, "growth": -0.1},                       # negative
        {k: v for k, v in SIGNAL_WEIGHTS.items() if k != "live"},  # missing
        {**SIGNAL_WEIGHTS, "vibes": 0.0},                          # unknown
        {k: 0.5 for k in SIGNALS},                                 # sum != 1
    ],
)
def test_bad_weights_rejected(weights):
    with pytest.raises(ValueError):
        validate_signal_weights(weights)


def test_valid_record_passes(full_record):
    assert validate_record(full_record) == []
    assert validate_record(full_record, strict_range=True) == []


def test_blank_id_rejected():
    rec = ArtistRecord(artist_id="  ", attributes={"reach": 0.5})
    assert any("artist_id" in p for p in validate_record(rec))


def test_empty_attributes_rejected():
    rec = ArtistRecord(artist_id="x", attributes={})
    assert any("empty" in p for p in validate_record(rec))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.5", None, True, 10**400])
def test_non_finite_or_non_numeric_values_rejected(bad):
    rec = ArtistRecord(artist_id="x", attributes={"reach": bad})
    assert any("reach" in p for p in validate_record(rec))


def test_unknown_attribute_rejected():
    rec = ArtistRecord(artist_id="x", attributes={"follower_count": 0.5})
    assert any("unknown attribute" in p for p in validate_record(rec))


def test_strict_range_flags_out_of_range_only_when_asked():
    rec = ArtistRecord(artist_id="x", attributes={"reach": 1.5})
    assert validate_record(rec) == []
    assert any("outside" in p for p in validate_record(rec, strict_range=True))


def test_duplicate_ids_flagged(full_record):
    twin = ArtistRecord(artist_id="full", attributes={"reach": 0.2})
    report = validate_records([full_record, twin])
    assert any("duplicate" in p for p in report.get("full", []))


def test_duplicate_ids_keep_both_records_problems():
    first = ArtistRecord(artist_id="dup", attributes={"reach": float("nan")})
    second = ArtistRecord(artist_id="dup", attributes={"reach": float("inf")})
    problems = validate_records([first, second])["dup"]
    assert sum("finite number" in p for p in problems) == 2  # neither swallowed
    assert any("duplicate" in p for p in problems)


def test_validate_records_clean_cohort(cohort):
    assert validate_records(cohort, strict_range=True) == {}


def test_coverage(full_record, sparse_record):
    assert schema.coverage(full_record) == 1.0
    assert 0.0 < schema.coverage(sparse_record) < 1.0
