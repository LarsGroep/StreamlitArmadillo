"""signals.py — both reduction modes yield ~5 well-behaved columns."""

import numpy as np
import pandas as pd
import pytest

from armadillo_scoring import signals
from armadillo_scoring.schema import SIGNALS, ArtistRecord


def test_manual_mode_yields_exactly_the_five_signals(cohort):
    sig = signals.to_signals(cohort, mode="manual")
    assert list(sig.columns) == list(SIGNALS)
    assert len(sig.columns) == 5
    assert set(sig.index) == {"full", "sparse", "mid"}


def test_pca_mode_yields_five_components(cohort):
    sig = signals.to_signals(cohort, mode="pca")
    # 3 artists -> centered matrix has rank <= 2, so the rank cap yields 2
    # components (a 3rd would be pure float noise min-max-blown to [0,1]).
    assert len(sig.columns) == len(cohort) - 1
    big = [
        ArtistRecord(f"a{i}", {a: (i * 7 + j * 3) % 10 / 10 for j, a in
                               enumerate(["rank_trend", "new_entry_rate", "chart_quality",
                                          "reach", "catalog_breadth", "recency",
                                          "recent_activity"])})
        for i in range(20)
    ]
    sig_big = signals.to_signals(big, mode="pca")
    assert list(sig_big.columns) == ["pc1", "pc2", "pc3", "pc4", "pc5"]
    assert "explained_variance_ratio" in sig_big.attrs
    assert len(sig_big.attrs["explained_variance_ratio"]) == 5


@pytest.mark.parametrize("mode", ["manual", "pca"])
def test_values_in_unit_range_and_no_nans(cohort, mode):
    sig = signals.to_signals(cohort, mode=mode)
    assert not sig.isna().any().any()
    assert (sig.to_numpy() >= 0).all() and (sig.to_numpy() <= 1).all()


def test_manual_averages_present_attributes_only(cohort):
    sig = signals.to_signals(cohort, mode="manual", fill_missing=False)
    # 'full' provides rank_trend=0.9 and new_entry_rate=0.8 -> growth = 0.85
    assert sig.loc["full", "growth"] == pytest.approx(0.85)
    # 'sparse' has only new_entry_rate=0.9 for growth -> growth = 0.9 (not /2)
    assert sig.loc["sparse", "growth"] == pytest.approx(0.9)
    # 'sparse' has no live attribute at all -> NaN before filling
    assert np.isnan(sig.loc["sparse", "live"])


def test_missing_signal_filled_with_population_median(cohort):
    sig = signals.to_signals(cohort, mode="manual")
    # only 'full' provides live_affinity (0.5); median of one value = 0.5
    assert sig.loc["sparse", "live"] == pytest.approx(0.5)


def test_all_missing_signal_falls_back_to_neutral():
    bare = [
        ArtistRecord("a", {"reach": 0.2, "recency": 0.4}),
        ArtistRecord("b", {"reach": 0.8, "recency": 0.6}),
    ]
    sig = signals.to_signals(bare, mode="manual")
    assert (sig["live"] == 0.5).all()
    assert (sig["growth"] == 0.5).all()


def test_pca_is_deterministic(cohort):
    a = signals.to_signals(cohort, mode="pca")
    b = signals.to_signals(cohort, mode="pca")
    pd.testing.assert_frame_equal(a, b)


def test_pca_components_oriented_with_attribute_level(cohort):
    """Contract anchoring: every pc must correlate non-negatively with the
    average attribute level, else positive score weights reward 'worse'."""
    attrs = signals.attribute_frame(cohort)
    filled = attrs.fillna(attrs.median()).dropna(axis=1, how="all")
    z = (filled - filled.mean()) / filled.std(ddof=0).replace(0.0, 1.0)
    row_mean = z.mean(axis=1)
    sig = signals.to_signals(cohort, mode="pca")
    for col in sig.columns:
        centered = sig[col] - sig[col].mean()
        assert float((centered * row_mean).sum()) >= -1e-9


def test_pca_no_noise_components_for_rank_deficient_input():
    """Two artists -> rank 1: exactly one real component, no [0,1] noise."""
    pair = [
        ArtistRecord("a", {"reach": 0.2, "recency": 0.4, "chart_quality": 0.3}),
        ArtistRecord("b", {"reach": 0.8, "recency": 0.6, "chart_quality": 0.9}),
    ]
    sig = signals.to_signals(pair, mode="pca")
    assert list(sig.columns) == ["pc1"]


def test_empty_attributes_record_kept_with_neutral_signals(cohort):
    """from_dict drops empty-dict keys; attribute_frame must restore them."""
    ghost = ArtistRecord("ghost", {})
    sig = signals.to_signals([*cohort, ghost], mode="manual")
    assert "ghost" in sig.index
    assert not sig.loc["ghost"].isna().any()  # median-filled, scoreable


def test_unknown_mode_rejected(cohort):
    with pytest.raises(ValueError, match="Unknown mode"):
        signals.to_signals(cohort, mode="shap")


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="empty"):
        signals.to_signals([])


def test_custom_signal_config_with_unknown_attribute_rejected(cohort):
    bad_config = {"growth": ("rank_trend", "instagram_followers")}
    with pytest.raises(ValueError, match="missing from the matrix"):
        signals.to_signals(cohort, mode="manual", signal_config=bad_config)
