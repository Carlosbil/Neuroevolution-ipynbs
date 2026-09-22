import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from neuroevolution.evaluation import cross_validation
from neuroevolution.evaluation.reporting import (
    build_held_out_results_rows,
    format_held_out_results_markdown,
    plot_fold_confusion_matrices,
)


def _fold_result(fold):
    return {
        "fold": fold,
        "accuracy": np.float64(80.0 + fold),
        "sensitivity": np.float64(70.0 + fold),
        "specificity": np.float64(75.0 + fold),
        "f1_score": np.float64(78.0 + fold),
        "auc": np.float64(82.0 + fold),
        "confusion_matrix": np.array([[4, 1], [1, 4]]),
        "n_samples": 10,
        "best_epoch": 2,
    }


def _aggregated_results():
    return {
        "fold_results": [_fold_result(1), _fold_result(2)],
        "mean_accuracy": 81.5,
        "std_accuracy": 0.5,
        "mean_sensitivity": 71.5,
        "std_sensitivity": 0.5,
        "mean_specificity": 76.5,
        "std_specificity": 0.5,
        "mean_f1": 79.5,
        "std_f1": 0.5,
        "mean_auc": 83.5,
        "std_auc": 0.5,
    }


def test_final_evaluation_persists_json_in_artifacts_with_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cross_validation,
        "load_fold_loaders",
        lambda *args, **kwargs: SimpleNamespace(train=[], validation=[], test=[]),
    )
    monkeypatch.setattr(
        cross_validation,
        "load_fold_test_loader",
        lambda *args, **kwargs: [],
    )
    calls = {"fold": 0}

    def fake_evaluate_single_fold(*args, **kwargs):
        calls["fold"] += 1
        return _fold_result(calls["fold"])

    monkeypatch.setattr(cross_validation, "evaluate_single_fold", fake_evaluate_single_fold)

    result = cross_validation.evaluate_5fold_cross_validation(
        best_genome={
            "num_conv_layers": 1,
            "num_fc_layers": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
        },
        config={
            "artifacts_dir": str(tmp_path),
            "num_epochs": 1,
            "data_path": str(tmp_path),
            "dataset_id": "synthetic_demo",
            "fold_id": "synthetic_demo",
            "fold_files_subdirectory": "synthetic",
            "final_evaluation_dataset_id": "real_demo",
            "final_evaluation_fold_id": "real_demo",
            "final_evaluation_fold_files_subdirectory": "real",
        },
        device=torch.device("cpu"),
    )

    assert result is not None
    assert result["selection_split"] == "synthetic_validation"
    assert result["internal_evaluation_split"] == "synthetic_test"
    assert result["evaluation_split"] == "real_test"
    assert result["training_data_source"] == "synthetic"
    assert result["final_evaluation_data_source"] == "real"
    assert result["real_data_used_for_weight_updates"] is False
    result_path = Path(result["results_path"])
    assert result_path.parent == tmp_path
    assert result_path.is_file()

    serialized = json.loads(result_path.read_text(encoding="utf-8"))
    assert serialized["selection_split"] == "synthetic_validation"
    assert serialized["evaluation_split"] == "real_test"
    assert serialized["training_dataset_id"] == "synthetic_demo"
    assert serialized["final_evaluation_dataset_id"] == "real_demo"
    assert serialized["fold_results"][0]["confusion_matrix"] == [[4, 1], [1, 4]]


def test_reporting_builds_complete_rows_and_one_plot_per_fold():
    results = _aggregated_results()

    rows = build_held_out_results_rows(results)
    assert [row["Fold"] for row in rows] == [1, 2, "Mean", "Std"]
    assert set(rows[0]) == {"Fold", "Accuracy", "Sensitivity", "Specificity", "F1", "AUC"}

    markdown = format_held_out_results_markdown(results)
    assert "| Mean | 81.50% | 71.50% | 76.50% | 79.50% | 83.50% |" in markdown

    figure = plot_fold_confusion_matrices(results)
    assert len(figure.axes) >= 2
    assert [axis.get_title() for axis in figure.axes[:2]] == [
        "Fold 1 — real held-out test",
        "Fold 2 — real held-out test",
    ]
