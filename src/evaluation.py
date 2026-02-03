"""Evaluation helpers for regression/classification, including time-series splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Split:
    train_slice: slice
    test_slice: slice


def time_train_test_split_index(n: int, test_size: float = 0.2) -> Split:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    split = int(n * (1 - test_size))
    return Split(slice(0, split), slice(split, n))


def walk_forward_splits(
    n: int,
    *,
    initial_train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
) -> Iterator[Split]:
    """Generate walk-forward train/test splits.

    Args:
        n: total number of samples
        initial_train_size: size of first training window
        test_size: size of each test window
        step_size: how far to advance each step (defaults to test_size)

    Yields:
        Split objects with train_slice/test_slice.
    """

    if initial_train_size <= 0:
        raise ValueError("initial_train_size must be positive")
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    step = test_size if step_size is None else step_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    train_end = initial_train_size
    while train_end + test_size <= n:
        yield Split(slice(0, train_end), slice(train_end, train_end + test_size))
        train_end += step


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    return out
