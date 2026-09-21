import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuroevolution.evaluation import cross_validation
from neuroevolution.evolution import fitness


def _write_tiny_fold(tmp_path, dataset_id="toy", fold_number=1):
    fold_dir = tmp_path / "folds"
    fold_dir.mkdir()

    arrays = {
        "train": (
            np.array([[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
        ),
        "val": (
            np.array(
                [
                    [2.0, 2.1, 2.2, 2.3],
                    [3.0, 3.1, 3.2, 3.3],
                    [4.0, 4.1, 4.2, 4.3],
                ],
                dtype=np.float32,
            ),
            np.array([1, 1, 0], dtype=np.int64),
        ),
        "test": (
            np.array(
                [
                    [5.0, 5.1, 5.2, 5.3],
                    [6.0, 6.1, 6.2, 6.3],
                    [7.0, 7.1, 7.2, 7.3],
                    [8.0, 8.1, 8.2, 8.3],
                ],
                dtype=np.float32,
            ),
            np.array([0, 0, 0, 1], dtype=np.int64),
        ),
    }

    for split_name, (x_values, y_values) in arrays.items():
        np.save(fold_dir / f"X_{split_name}_{dataset_id}_fold_{fold_number}.npy", x_values)
        np.save(fold_dir / f"y_{split_name}_{dataset_id}_fold_{fold_number}.npy", y_values)

    return {
        "data_path": str(tmp_path),
        "fold_files_subdirectory": "folds",
        "fold_id": "unused",
        "dataset_id": dataset_id,
        "batch_size": 16,
        "fold_cache_mode": "none",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
    }


def _loader_labels(loader):
    labels = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.tolist())
    return labels


def _one_batch_loader(labels):
    x_values = torch.zeros(len(labels), 1, 4)
    y_values = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x_values, y_values), batch_size=max(1, len(labels)))


def test_fold_loader_validation_split_excludes_test_samples(tmp_path):
    config = _write_tiny_fold(tmp_path)
    device = torch.device("cpu")

    train_loader, validation_loader = fitness.load_fold_data(
        1,
        config,
        device,
        eval_split="validation",
    )

    assert len(train_loader.dataset) == 2
    assert len(validation_loader.dataset) == 3
    assert _loader_labels(validation_loader) == [1, 1, 0]


def test_fold_loader_can_explicitly_load_test_split(tmp_path):
    config = _write_tiny_fold(tmp_path)
    device = torch.device("cpu")

    _, test_loader = fitness.load_fold_data(1, config, device, eval_split="test")

    assert len(test_loader.dataset) == 4
    assert _loader_labels(test_loader) == [0, 0, 0, 1]


def test_fold_loader_cache_key_includes_eval_split(tmp_path):
    config = _write_tiny_fold(tmp_path)
    device = torch.device("cpu")

    validation_key = fitness._build_fold_cache_key(1, config, device, "ram", "validation")
    test_key = fitness._build_fold_cache_key(1, config, device, "ram", "test")

    assert validation_key != test_key


def test_train_fold_requests_validation_only_loader(monkeypatch):
    requested_splits = []

    def fake_load_fold_data(fold_num, config, device, eval_split="validation"):
        requested_splits.append(eval_split)
        loader = _one_batch_loader([0])
        return loader, loader

    def invalid_model(*args, **kwargs):
        raise ValueError("Invalid architecture for test")

    monkeypatch.setattr(fitness, "load_fold_data", fake_load_fold_data)
    monkeypatch.setattr(fitness, "EvolvableCNN", invalid_model)

    result = fitness.train_fold_in_thread(
        {"optimizer": "adam", "learning_rate": 0.001},
        1,
        {"batch_size": 1},
        torch.device("cpu"),
    )

    assert requested_splits == ["validation"]
    assert result[1] == 0.0


def test_evaluate_fitness_marks_aggregated_metrics_as_validation(monkeypatch):
    def fake_train_fold(genome, fold_num, config, device):
        return (
            fold_num,
            float(10 * fold_num),
            f"model-{fold_num}",
            {
                "accuracy": 10.0,
                "sensitivity": 20.0,
                "specificity": 30.0,
                "precision": 40.0,
                "f1_score": float(10 * fold_num),
                "auc": 50.0,
                "evaluation_split": "validation",
            },
        )

    monkeypatch.setattr(fitness, "train_fold_in_thread", fake_train_fold)

    fitness_score, best_model, metrics = fitness.evaluate_fitness(
        {"id": "g1"},
        {"num_folds": 2, "fold_parallel_workers": 1},
        torch.device("cpu"),
    )

    assert fitness_score == pytest.approx(15.0)
    assert best_model == "model-2"
    assert metrics["fitness_split"] == "validation"
    assert {m["evaluation_split"] for m in metrics["fold_metrics"].values()} == {"validation"}


def test_final_loader_requests_all_splits(monkeypatch):
    requested_splits = []

    def fake_evolution_loader(fold_number, config, device, eval_split=None):
        requested_splits.append(eval_split)
        if eval_split == "all":
            return "train-loader", "validation-loader", "test-loader"
        return "unexpected-train", "unexpected-eval"

    monkeypatch.setattr(cross_validation, "load_fold_data_from_evolution", fake_evolution_loader)

    loaded = cross_validation.load_fold_data({}, 1, device=torch.device("cpu"))

    assert loaded == ("train-loader", "validation-loader", "test-loader")
    assert requested_splits == ["all"]


def test_final_evaluation_reports_test_metrics_after_validation_selection(monkeypatch):
    class AlwaysClassZeroModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.logits = nn.Parameter(torch.tensor([1.0, 0.0]))

        def forward(self, data):
            return self.logits.expand(data.size(0), 2)

    class NoOpOptimizer:
        def __init__(self, params, lr):
            self.params = list(params)

        def zero_grad(self):
            for param in self.params:
                param.grad = None

        def step(self):
            return None

    monkeypatch.setattr(cross_validation, "EvolvableCNN", AlwaysClassZeroModel)
    monkeypatch.setitem(cross_validation.OPTIMIZERS, "adam", NoOpOptimizer)

    result = cross_validation.evaluate_single_fold(
        {"optimizer": "adam", "learning_rate": 0.001},
        {"epoch_patience": 5, "improvement_threshold": 0.0},
        _one_batch_loader([0]),
        _one_batch_loader([0, 0]),
        _one_batch_loader([1, 1, 1]),
        1,
        torch.device("cpu"),
        num_epochs=1,
    )

    assert result["selection_split"] == "validation"
    assert result["evaluation_split"] == "test"
    assert result["best_validation_acc"] == pytest.approx(100.0)
    assert result["accuracy"] == pytest.approx(0.0)
    assert result["n_samples"] == 3
