"""Model evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_predictions(y_true, y_pred) -> tuple[dict[str, float], pd.DataFrame]:
    """Return summary metrics and a detailed classification report."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    report = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).transpose()
    return metrics, report


def save_confusion_matrix(y_true, y_pred, output: Path) -> None:
    """Save a labeled confusion-matrix figure."""
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_xlabel("Predicted imbalance (g)")
    axis.set_ylabel("True imbalance (g)")
    axis.set_title("Motor imbalance confusion matrix")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
