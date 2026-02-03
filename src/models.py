"""Model training and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class SplitData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def time_train_test_split(x: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> SplitData:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    split = int(len(x) * (1 - test_size))
    return SplitData(x[:split], x[split:], y[:split], y[split:])


def train_linear_regression(x_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def train_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    return model


def train_lasso(x_train: np.ndarray, y_train: np.ndarray, alpha: float = 0.1) -> Lasso:
    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(x_train, y_train)
    return model


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred),
    }


def train_logistic(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=5000)
    model.fit(x_train, y_train)
    return model


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
