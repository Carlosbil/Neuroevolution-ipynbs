"""Evaluation package - metrics calculation and artifact management."""

from .metrics import calculate_metrics, aggregate_fold_metrics
from .artifacts import ArtifactManager
from .cross_validation import evaluate_single_fold, evaluate_5fold_cross_validation
from .reporting import (
    build_held_out_results_rows,
    format_held_out_results_markdown,
    plot_fold_confusion_matrices,
)

__all__ = [
    'calculate_metrics',
    'aggregate_fold_metrics',
    'ArtifactManager',
    'evaluate_single_fold',
    'evaluate_5fold_cross_validation',
    'build_held_out_results_rows',
    'format_held_out_results_markdown',
    'plot_fold_confusion_matrices',
]
