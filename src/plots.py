"""Plotting helpers for notebooks."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_series(df, columns: Sequence[str], title: str) -> None:
    ax = df[columns].plot(figsize=(10, 4))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    plt.tight_layout()


def plot_residuals(y_true, y_pred, title: str = "Residuals") -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title(title)
    plt.xlabel("Residual")
    plt.tight_layout()


def plot_confusion_matrix(cm, labels) -> None:
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()


def plot_roc_curve(fpr, tpr) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()


def plot_pr_curve(recall, precision) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()


def plot_coefficients(
    df,
    *,
    feature_col: str = "feature",
    coef_col: str = "coef",
    top_n: int = 15,
    title: str = "Top Coefficients",
) -> None:
    """Horizontal bar plot for coefficient tables."""

    tmp = df[[feature_col, coef_col]].copy().head(top_n)
    plt.figure(figsize=(8, max(4, int(top_n * 0.35))))
    sns.barplot(data=tmp, y=feature_col, x=coef_col, orient="h")
    plt.title(title)
    plt.tight_layout()
