"""
Metrics calculation module.

Provides functions to calculate classification metrics including:
- Accuracy, Sensitivity (Recall), Specificity, Precision
- F1-Score, AUC (Area Under ROC Curve)
- Confusion matrix components (TP, TN, FP, FN)
"""

import numpy as np
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate classification metrics from true labels and predictions.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_probs: Predicted probabilities for class 1 (optional, for AUC)

    Returns:
        Dictionary with all metrics as percentages (0-100)
    """
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Basic metrics
    accuracy = 100.0 * (tp + tn) / max(1, len(y_true))
    sensitivity = 100.0 * tp / max(1, tp + fn)  # Also known as Recall
    specificity = 100.0 * tn / max(1, tn + fp)
    precision = 100.0 * tp / max(1, tp + fp)
    
    # F1-Score (harmonic mean of precision and recall)
    f1_score = 2.0 * precision * sensitivity / max(1e-8, precision + sensitivity)

    # AUC (Area Under ROC Curve)
    auc = 0.0
    if y_probs is not None and len(np.unique(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, y_probs) * 100.0)
        except Exception:
            auc = 0.0

    return {
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1_score),
        'auc': float(auc),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def aggregate_fold_metrics(fold_metrics: list) -> Dict[str, float]:
    """
    Aggregate metrics from multiple folds (e.g., 5-fold CV).

    Args:
        fold_metrics: List of metric dictionaries from each fold

    Returns:
        Dictionary with mean and std for each metric
    """
    if not fold_metrics:
        return {
            'accuracy': 0.0, 'accuracy_std': 0.0,
            'sensitivity': 0.0, 'sensitivity_std': 0.0,
            'specificity': 0.0, 'specificity_std': 0.0,
            'precision': 0.0, 'precision_std': 0.0,
            'f1_score': 0.0, 'f1_score_std': 0.0,
            'auc': 0.0, 'auc_std': 0.0,
            'n_folds': 0
        }

    valid_metrics = [m for m in fold_metrics if m is not None]
    
    if not valid_metrics:
        return {
            'accuracy': 0.0, 'accuracy_std': 0.0,
            'sensitivity': 0.0, 'sensitivity_std': 0.0,
            'specificity': 0.0, 'specificity_std': 0.0,
            'precision': 0.0, 'precision_std': 0.0,
            'f1_score': 0.0, 'f1_score_std': 0.0,
            'auc': 0.0, 'auc_std': 0.0,
            'n_folds': 0
        }

    return {
        'accuracy': np.mean([m['accuracy'] for m in valid_metrics]),
        'accuracy_std': np.std([m['accuracy'] for m in valid_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in valid_metrics]),
        'sensitivity_std': np.std([m['sensitivity'] for m in valid_metrics]),
        'specificity': np.mean([m['specificity'] for m in valid_metrics]),
        'specificity_std': np.std([m['specificity'] for m in valid_metrics]),
        'precision': np.mean([m['precision'] for m in valid_metrics]),
        'precision_std': np.std([m['precision'] for m in valid_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in valid_metrics]),
        'f1_score_std': np.std([m['f1_score'] for m in valid_metrics]),
        'auc': np.mean([m['auc'] for m in valid_metrics]),
        'auc_std': np.std([m['auc'] for m in valid_metrics]),
        'n_folds': len(valid_metrics)
    }
