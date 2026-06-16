"""score.py — deterministic weighted score whose breakdown adds up exactly."""

import numpy as np
import pandas as pd
import pytest

from armadillo_scoring import score, signals
from armadillo_scoring.schema import SIGNAL_WEIGHTS, SIGNALS


@pytest.fixture
def signal_matrix(cohort):
    return signals.to_signals(cohort, mode="manual")


def test_score_is_deterministic(signal_matrix):
    a = score.speed_score(signal_matrix)
    b = score.speed_score(signal_matrix.copy())
    pd.testing.assert_frame_equal(a, b)


def test_contributions_sum_to_score_exactly(signal_matrix):
    scored = score.speed_score(signal_matrix)
    contrib_cols = [c for c in scored.columns if c.startswith(score.CONTRIB_PREFIX)]
    assert len(contrib_cols) == len(SIGNALS)
    np.testing.assert_allclose(
        scored[contrib_cols].sum(axis=1), scored[score.SCORE_COL], rtol=0, atol=1e-12
    )


def test_score_in_unit_range(signal_matrix):
    scored = score.speed_score(signal_matrix)
    assert (scored[score.SCORE_COL] >= 0).all()
    assert (scored[score.SCORE_COL] <= 1).all()


def test_default_weights_used_for_manual_signals(signal_matrix):
    scored = score.speed_score(signal_matrix)
    assert scored.attrs["weights"] == pytest.approx(SIGNAL_WEIGHTS)


def test_explicit_weights_override_and_renormalize(signal_matrix):
    only_growth = {s: (1.0 if s == "growth" else 0.0) for s in SIGNALS}
    scored = score.speed_score(signal_matrix, weights=only_growth)
    np.testing.assert_allclose(
        scored[score.SCORE_COL], signal_matrix.loc[scored.index, "growth"], atol=1e-12
    )
    # un-normalized input weights get scaled to sum 1
    doubled = {s: 2 * w for s, w in SIGNAL_WEIGHTS.items()}
    assert score.speed_score(signal_matrix, weights=doubled).attrs["weights"] == pytest.approx(
        SIGNAL_WEIGHTS
    )


def test_missing_weight_entry_rejected(signal_matrix):
    with pytest.raises(ValueError, match="missing entries"):
        score.speed_score(signal_matrix, weights={"growth": 1.0})


def test_negative_weights_rejected(signal_matrix):
    bad = {**SIGNAL_WEIGHTS, "growth": -0.25, "live": 0.6}
    with pytest.raises(ValueError, match="non-negative"):
        score.speed_score(signal_matrix, weights=bad)


def test_nan_signals_rejected(signal_matrix):
    dirty = signal_matrix.copy()
    dirty.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        score.speed_score(dirty)


def test_pca_signals_use_explained_variance_weights(cohort):
    sig = signals.to_signals(cohort, mode="pca")
    scored = score.speed_score(sig)
    ratios = sig.attrs["explained_variance_ratio"]
    expected = {c: r / sum(ratios) for c, r in zip(sig.columns, ratios)}
    assert scored.attrs["weights"] == pytest.approx(expected)


def test_output_sorted_best_first_with_stable_ties():
    sig = pd.DataFrame(
        {s: [0.4, 0.4, 0.8] for s in SIGNALS},
        index=["zeta", "alpha", "top"],
    )
    scored = score.speed_score(sig)
    assert list(scored.index) == ["top", "alpha", "zeta"]  # tie -> id order


def test_equal_weights_fallback_warns_on_unrecognized_columns():
    sig = pd.DataFrame({s: [0.4, 0.8] for s in SIGNALS if s != "live"},
                       index=["a", "b"])
    with pytest.warns(UserWarning, match="equal weights"):
        scored = score.speed_score(sig)
    n = len(sig.columns)
    assert scored.attrs["weights"] == pytest.approx({c: 1.0 / n for c in sig.columns})


def test_mismatched_explained_variance_ratio_rejected():
    sig = pd.DataFrame({"pc1": [0.1, 0.9], "pc2": [0.2, 0.8]}, index=["a", "b"])
    sig.attrs["explained_variance_ratio"] = [1.0]  # wrong length vs 2 columns
    with pytest.raises(ValueError, match="does not match"):
        score.speed_score(sig)


def test_explain_mentions_every_signal(signal_matrix):
    scored = score.speed_score(signal_matrix)
    text = score.explain(scored, "full")
    assert "full" in text and "speed_score" in text
    for s in SIGNALS:
        assert s in text


def test_explain_arithmetic_matches_weights_and_values(signal_matrix):
    scored = score.speed_score(signal_matrix)
    text = score.explain(scored, "full")
    for s in SIGNALS:
        weight = scored.attrs["weights"][s]
        value = signal_matrix.loc["full", s]
        assert f"weight {weight:.2f} x value {value:.2f}" in text


def test_explain_survives_missing_weights_attrs(signal_matrix):
    scored = score.speed_score(signal_matrix)
    scored.attrs = {}  # attrs don't survive e.g. serialization round-trips
    text = score.explain(scored, "full")
    assert "speed_score" in text
    for s in SIGNALS:
        assert s in text


def test_explain_unknown_artist_rejected(signal_matrix):
    scored = score.speed_score(signal_matrix)
    with pytest.raises(KeyError):
        score.explain(scored, "nobody")
