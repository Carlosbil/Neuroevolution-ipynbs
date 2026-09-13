"""Presentation helpers for final held-out evaluation results."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


METRIC_COLUMNS: Sequence[tuple[str, str]] = (
    ("Accuracy", "accuracy"),
    ("Sensitivity", "sensitivity"),
    ("Specificity", "specificity"),
    ("F1", "f1_score"),
    ("AUC", "auc"),
)


def build_held_out_results_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build per-fold plus aggregate rows without mixing validation metrics."""
    rows = []
    for fold_result in results.get("fold_results", []):
        row = {"Fold": fold_result["fold"]}
        row.update({label: fold_result[key] for label, key in METRIC_COLUMNS})
        rows.append(row)

    for label, prefix in (("Mean", "mean"), ("Std", "std")):
        row = {"Fold": label}
        row.update({metric_label: results[f"{prefix}_{metric_key}"] for metric_label, metric_key in METRIC_COLUMNS})
        rows.append(row)
    return rows


def format_held_out_results_markdown(results: Dict[str, Any]) -> str:
    """Format final held-out metrics as a Markdown table with percentage values."""
    headers = ["Fold", *(label for label, _ in METRIC_COLUMNS)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in build_held_out_results_rows(results):
        values = [str(row["Fold"]), *(f"{row[label]:.2f}%" for label, _ in METRIC_COLUMNS)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_fold_confusion_matrices(results: Dict[str, Any]):
    """Return a labelled confusion-matrix figure for every successful test fold."""
    import matplotlib.pyplot as plt

    fold_results = results.get("fold_results", [])
    if not fold_results:
        raise ValueError("No held-out fold results are available to plot.")

    figure, axes = plt.subplots(1, len(fold_results), figsize=(5 * len(fold_results), 4), squeeze=False)
    for axis, fold_result in zip(axes[0], fold_results):
        matrix = np.asarray(fold_result["confusion_matrix"])
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set_title(f"Fold {fold_result['fold']} — held-out test")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(matrix.shape[1]))
        axis.set_yticks(range(matrix.shape[0]))
        threshold = matrix.max() / 2 if matrix.size else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                )
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return figure
