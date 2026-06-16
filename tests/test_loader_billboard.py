"""loaders/billboard.py — integration tests against the real public CSV,
plus unit tests on a synthetic mini-CSV."""

import textwrap

import pytest

from armadillo_scoring import schema, score, signals, validate
from armadillo_scoring.loaders import billboard


# --------------------------------------------------------------------------- #
# Synthetic mini-CSV: exact, hand-checkable expectations.
# --------------------------------------------------------------------------- #
@pytest.fixture
def mini_csv(tmp_path):
    path = tmp_path / "mini.csv"
    path.write_text(
        textwrap.dedent(
            """\
            ranking,song,band_singer,year,danceability,energy,valence,liveness
            1,Hit One,Riser,2020,,,,
            5,Hit Two,Riser,2022,,,,
            90,Slow Song,OneShot,2010,0.5,0.7,0.9,0.3
            """
        )
    )
    return path


def test_mini_csv_mapping(mini_csv):
    records = {r.artist_id: r for r in billboard.load(mini_csv)}
    assert set(records) == {"Riser", "OneShot"}

    riser, oneshot = records["Riser"], records["OneShot"]

    # chart_quality: inverted rank mean. Riser: ((101-1)+(101-5))/2/100 = 0.98
    assert riser.attributes["chart_quality"] == pytest.approx(0.98)
    assert oneshot.attributes["chart_quality"] == pytest.approx(0.11)

    # Riser charted in two distinct years -> has a trend; OneShot doesn't.
    assert "rank_trend" in riser.attributes
    assert "rank_trend" not in oneshot.attributes

    # OneShot has audio rows -> appeal = mean(0.5, 0.7, 0.9) = 0.7, live = 0.3
    assert oneshot.attributes["content_appeal"] == pytest.approx(0.7)
    assert oneshot.attributes["live_affinity"] == pytest.approx(0.3)
    assert "content_appeal" not in riser.attributes

    # recency: last year mapped over the 2010-2022 span
    assert riser.attributes["recency"] == pytest.approx(1.0)
    assert oneshot.attributes["recency"] == pytest.approx(0.0)

    # meta provenance survives
    assert riser.meta["best_rank"] == 1
    assert riser.meta["entries"] == 2


def test_missing_required_column_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("song,year\nA,2000\n")
    with pytest.raises(ValueError, match="missing required columns"):
        billboard.load(bad)


def test_partial_audio_columns_emit_only_computable_attributes(tmp_path):
    """Appeal columns without a liveness column must not crash, and must
    emit content_appeal while omitting live_affinity (never inventing it)."""
    path = tmp_path / "partial.csv"
    path.write_text(
        "ranking,song,band_singer,year,danceability,energy,valence\n"
        "10,A,Artist,2020,0.4,0.6,0.8\n"
    )
    (record,) = billboard.load(path)
    assert record.attributes["content_appeal"] == pytest.approx(0.6)
    assert "live_affinity" not in record.attributes


def test_artist_literally_named_na_is_kept(tmp_path):
    """pandas default NA parsing would silently drop a band named 'NA'."""
    path = tmp_path / "na.csv"
    path.write_text(
        "ranking,song,band_singer,year\n"
        "10,Song A,NA,2020\n"
        "20,Song B,Real Artist,2020\n"
    )
    ids = {r.artist_id for r in billboard.load(path)}
    assert ids == {"NA", "Real Artist"}


# --------------------------------------------------------------------------- #
# Real public dataset: the loader's output must satisfy the schema contract.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_records():
    if not billboard.DEFAULT_CSV.exists():
        pytest.skip("public billboard CSV not present")
    return billboard.load()


def test_real_csv_loads_all_artists(real_records):
    assert len(real_records) > 1000  # 1039 in the 24-year file


def test_real_records_pass_schema_validation(real_records):
    assert schema.validate_records(real_records, strict_range=True) == {}


def test_real_records_have_unique_ids(real_records):
    ids = [r.artist_id for r in real_records]
    assert len(ids) == len(set(ids))


def test_real_load_is_deterministic(real_records):
    again = billboard.load()
    assert [r.artist_id for r in again] == [r.artist_id for r in real_records]
    assert all(
        a.attributes == b.attributes for a, b in zip(again, real_records)
    )


def test_audio_attributes_are_optional_not_invented(real_records):
    with_audio = [r for r in real_records if "content_appeal" in r.attributes]
    without = [r for r in real_records if "content_appeal" not in r.attributes]
    # the public file only has audio for ~2000-2004 -> both groups must exist
    assert with_audio and without


# --------------------------------------------------------------------------- #
# End-to-end: pins the README results table — if these numbers move, the
# README must be updated in the same commit (the pipeline is deterministic).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_labels():
    if not billboard.DEFAULT_CSV.exists():
        pytest.skip("public billboard CSV not present")
    raw = billboard.load_raw()
    return validate.hit_labels_from_chart(
        raw, artist_col=billboard.ARTIST_COL, rank_col=billboard.RANK_COL, top_rank=10
    )


@pytest.mark.parametrize("mode,expected_auc", [("manual", 0.845), ("pca", 0.895)])
def test_end_to_end_pipeline_matches_readme(real_records, real_labels, mode, expected_auc):
    sig = signals.to_signals(real_records, mode=mode)
    scored = score.speed_score(sig)
    report = validate.evaluate(scored[score.SCORE_COL], real_labels)
    assert report.n_artists == 1039
    assert report.base_rate == pytest.approx(0.215, abs=0.005)
    assert report.auc == pytest.approx(expected_auc, abs=0.01)
    assert report.precision_at_k[10] >= 0.7  # discriminative-ranking floor
