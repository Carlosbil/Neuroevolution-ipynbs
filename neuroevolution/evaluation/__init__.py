"""Evaluation package - metrics calculation and artifact management."""

from .metrics import calculate_metrics, aggregate_fold_metrics
from .artifacts import ArtifactManager
from .cross_validation import evaluate_single_fold, evaluate_5fold_cross_validation

__all__ = [
    'calculate_metrics',
    'aggregate_fold_metrics',
    'ArtifactManager',
    'evaluate_single_fold',
    'evaluate_5fold_cross_validation',
]
