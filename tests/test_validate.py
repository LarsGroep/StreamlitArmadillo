"""validate.py — proxy labels and separation metrics."""

import pandas as pd
import pytest

from armadillo_scoring import validate


@pytest.fixture
def chart():
    """Tiny chart: artist 'big' peaked at 3 (hit), 'mid' at 40, 'tail' at 95."""
    return pd.DataFrame(
        {
            "artist": ["big", "big", "mid", "tail", "mid"],
            "rank": [3, 50, 40, 95, 60],
        }
    )


def test_hit_labels_top_rank(chart):
    labels = validate.hit_labels_from_chart(chart, artist_col="artist", rank_col="rank")
    assert labels.to_dict() == {"big": 1, "mid": 0, "tail": 0}


def test_hit_labels_top_fraction(chart):
    labels = validate.hit_labels_from_chart(
        chart, artist_col="artist", rank_col="rank", top_rank=None, top_fraction=0.34
    )
    assert labels["big"] == 1
    assert labels["tail"] == 0


def test_hit_labels_requires_exactly_one_mode(chart):
    with pytest.raises(ValueError):
        validate.hit_labels_from_chart(
            chart, artist_col="artist", rank_col="rank", top_rank=10, top_fraction=0.1
        )
    with pytest.raises(ValueError):
        validate.hit_labels_from_chart(
            chart, artist_col="artist", rank_col="rank", top_rank=None, top_fraction=None
        )


def test_auc_perfect_separation():
    scores = pd.Series({"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1})
    labels = pd.Series({"a": 1, "b": 1, "c": 0, "d": 0})
    assert validate.auc(scores, labels) == 1.0


def test_auc_inverted_scores_give_zero():
    scores = pd.Series({"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9})
    labels = pd.Series({"a": 1, "b": 1, "c": 0, "d": 0})
    assert validate.auc(scores, labels) == 0.0


def test_auc_all_tied_is_half():
    scores = pd.Series({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5})
    labels = pd.Series({"a": 1, "b": 1, "c": 0, "d": 0})
    assert validate.auc(scores, labels) == 0.5


def test_auc_single_class_rejected():
    scores = pd.Series({"a": 0.9, "b": 0.1})
    with pytest.raises(ValueError, match="both classes"):
        validate.auc(scores, pd.Series({"a": 1, "b": 1}))


def test_auc_aligns_on_index_and_drops_unmatched():
    scores = pd.Series({"a": 0.9, "b": 0.1, "ghost": 0.99})
    labels = pd.Series({"a": 1, "b": 0, "other_ghost": 1})
    assert validate.auc(scores, labels) == 1.0  # ghosts dropped


def test_precision_at_k():
    scores = pd.Series({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6})
    labels = pd.Series({"a": 1, "b": 0, "c": 1, "d": 0})
    assert validate.precision_at_k(scores, labels, 1) == 1.0
    assert validate.precision_at_k(scores, labels, 2) == 0.5
    assert validate.precision_at_k(scores, labels, 4) == 0.5
    # k larger than cohort: clamps to cohort size
    assert validate.precision_at_k(scores, labels, 100) == 0.5


def test_precision_at_k_tie_break_is_deterministic():
    scores = pd.Series({"zeta": 0.5, "alpha": 0.5})
    labels = pd.Series({"zeta": 0, "alpha": 1})
    # tie broken by id: 'alpha' first -> hit
    assert validate.precision_at_k(scores, labels, 1) == 1.0


def test_precision_at_k_rejects_bad_k():
    scores = pd.Series({"a": 0.9})
    labels = pd.Series({"a": 1})
    with pytest.raises(ValueError):
        validate.precision_at_k(scores, labels, 0)


def test_precision_at_k_rejects_disjoint_indices():
    with pytest.raises(ValueError, match="share no artists"):
        validate.precision_at_k(pd.Series({"a": 0.9}), pd.Series({"z": 1}), 5)


def test_evaluate_report():
    scores = pd.Series({"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1})
    labels = {"a": 1, "b": 1, "c": 0, "d": 0}
    report = validate.evaluate(scores, labels, ks=(2,))
    assert report.auc == 1.0
    assert report.precision_at_k[2] == 1.0
    assert report.n_artists == 4
    assert report.n_hits == 2
    assert report.base_rate == 0.5
    text = str(report)
    assert "AUC" in text and "precision@2" in text
