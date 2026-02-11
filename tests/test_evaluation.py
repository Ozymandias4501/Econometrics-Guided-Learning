import numpy as np
import pytest

from src import evaluation
from src.evaluation import Split


# ── time_train_test_split_index ──────────────────────────────────────────

def test_split_sizes():
    sp = evaluation.time_train_test_split_index(100, test_size=0.2)
    assert sp.train_slice == slice(0, 80)
    assert sp.test_slice == slice(80, 100)


def test_split_invalid_test_size():
    with pytest.raises(ValueError):
        evaluation.time_train_test_split_index(100, test_size=0.0)
    with pytest.raises(ValueError):
        evaluation.time_train_test_split_index(100, test_size=1.0)


def test_split_no_overlap():
    sp = evaluation.time_train_test_split_index(200, test_size=0.3)
    assert sp.train_slice.stop == sp.test_slice.start


# ── walk_forward_splits ──────────────────────────────────────────────────

def test_walk_forward_count():
    splits = list(evaluation.walk_forward_splits(100, initial_train_size=50, test_size=10))
    assert len(splits) == 5  # 50, 60, 70, 80, 90


def test_walk_forward_expanding_train():
    splits = list(evaluation.walk_forward_splits(100, initial_train_size=50, test_size=10))
    for i, sp in enumerate(splits):
        assert sp.train_slice.start == 0
        assert sp.train_slice.stop == 50 + i * 10


def test_walk_forward_no_train_test_overlap():
    splits = list(evaluation.walk_forward_splits(200, initial_train_size=80, test_size=20))
    for sp in splits:
        assert sp.train_slice.stop <= sp.test_slice.start


def test_walk_forward_custom_step():
    splits = list(evaluation.walk_forward_splits(
        100, initial_train_size=50, test_size=10, step_size=5
    ))
    assert len(splits) >= 5
    for sp in splits:
        assert sp.train_slice.stop <= sp.test_slice.start


def test_walk_forward_rejects_bad_inputs():
    with pytest.raises(ValueError):
        list(evaluation.walk_forward_splits(100, initial_train_size=0, test_size=10))
    with pytest.raises(ValueError):
        list(evaluation.walk_forward_splits(100, initial_train_size=50, test_size=0))
    with pytest.raises(ValueError):
        list(evaluation.walk_forward_splits(100, initial_train_size=50, test_size=10, step_size=-1))


# ── regression_metrics ───────────────────────────────────────────────────

def test_regression_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = evaluation.regression_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_regression_metrics_keys():
    y = np.array([1.0, 2.0, 3.0])
    m = evaluation.regression_metrics(y, y + 0.1)
    assert set(m.keys()) == {"mae", "rmse", "r2"}


# ── classification_metrics ───────────────────────────────────────────────

def test_classification_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluation.classification_metrics(y_true, y_prob)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_classification_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.3, 0.7, 0.4, 0.6])
    m = evaluation.classification_metrics(y_true, y_prob)
    expected = {"threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"}
    assert set(m.keys()) == expected


def test_classification_metrics_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.3, 0.4, 0.6, 0.7])
    m_low = evaluation.classification_metrics(y_true, y_prob, threshold=0.35)
    m_high = evaluation.classification_metrics(y_true, y_prob, threshold=0.65)
    assert m_low["recall"] >= m_high["recall"]
    assert m_low["threshold"] == pytest.approx(0.35)
