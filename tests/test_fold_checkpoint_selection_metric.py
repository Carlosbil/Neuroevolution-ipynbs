import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuroevolution.config import get_default_config, validate_config
from neuroevolution.evaluation import cross_validation
from neuroevolution.evolution import fitness


def _one_batch_loader(labels):
    x_values = torch.zeros(len(labels), 1, 4)
    y_values = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x_values, y_values), batch_size=max(1, len(labels)))


class EpochPreferenceModel(nn.Module):
    """Epoch 1 predicts class 0; later epochs predict class 1."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.train_logits = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.register_buffer("epoch_marker", torch.tensor(0.0))

    def forward(self, data):
        if self.training:
            with torch.no_grad():
                self.epoch_marker.add_(1.0)
            return self.train_logits.expand(data.size(0), 2)

        if int(self.epoch_marker.item()) <= 1:
            logits = torch.tensor([2.0, 0.0], device=data.device)
        else:
            logits = torch.tensor([0.0, 2.0], device=data.device)
        return logits.expand(data.size(0), 2)


class NoOpOptimizer:
    def __init__(self, params, lr):
        self.params = list(params)

    def zero_grad(self, *args, **kwargs):
        for param in self.params:
            param.grad = None

    def step(self):
        return None


def _selection_config(metric_name):
    return {
        "batch_size": 10,
        "num_epochs": 2,
        "early_stopping_patience": 1,
        "epoch_patience": 5,
        "improvement_threshold": 0.0,
        "metric_improvement_threshold": 0.0,
        "validation_frequency_epochs": 1,
        "fold_selection_metric": metric_name,
        "fitness_metric": "f1_score",
        "use_amp": False,
    }


def _patch_epoch_preference_training(monkeypatch):
    train_loader = _one_batch_loader([0])
    validation_loader = _one_batch_loader([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    def fake_load_fold_data(fold_num, config, device, eval_split="validation"):
        return train_loader, validation_loader

    monkeypatch.setattr(fitness, "load_fold_data", fake_load_fold_data)
    monkeypatch.setattr(fitness, "EvolvableCNN", EpochPreferenceModel)
    monkeypatch.setitem(fitness.OPTIMIZERS, "adam", NoOpOptimizer)
    return train_loader, validation_loader


def test_default_config_validates_metric_selection_defaults():
    config = get_default_config()

    assert config["fitness_metric"] == "f1_score"
    assert config["fold_selection_metric"] == "fitness_metric"
    assert config["metric_improvement_threshold"] is None
    validate_config(config)


def test_validate_config_rejects_unknown_metric_names():
    config = get_default_config()
    config["fold_selection_metric"] = "balanced_accuracy"

    with pytest.raises(ValueError, match="fold_selection_metric"):
        validate_config(config)

    config = get_default_config()
    config["fitness_metric"] = "balanced_accuracy"

    with pytest.raises(ValueError, match="fitness_metric"):
        validate_config(config)


def test_metric_helpers_resolve_aliases_and_compute_binary_metrics():
    config = {"fitness_metric": "f1_score", "fold_selection_metric": "fitness_metric"}

    assert fitness.resolve_metric_name(config, "fold_selection_metric") == "f1_score"
    assert fitness.resolve_metric_name({"fitness_metric": "recall"}, "fitness_metric") == "sensitivity"
    assert fitness.metric_value({"sensitivity": 42.0}, "recall") == pytest.approx(42.0)
    assert fitness.metric_value({"f1_score": None}, "f1_score") == pytest.approx(0.0)

    metrics = fitness.compute_classification_metrics(
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0.9, 0.4, 0.2, 0.8],
    )

    assert metrics["accuracy"] == pytest.approx(50.0)
    assert metrics["sensitivity"] == pytest.approx(50.0)
    assert metrics["specificity"] == pytest.approx(50.0)
    assert metrics["precision"] == pytest.approx(50.0)
    assert metrics["f1_score"] == pytest.approx(50.0)
    assert metrics["auc"] == pytest.approx(75.0)


def test_train_fold_selects_f1_best_epoch_when_accuracy_prefers_another(monkeypatch):
    _patch_epoch_preference_training(monkeypatch)

    fold_num, fold_score, model, metrics = fitness.train_fold_in_thread(
        {"optimizer": "adam", "learning_rate": 0.001},
        1,
        _selection_config("f1_score"),
        torch.device("cpu"),
    )

    assert fold_num == 1
    assert int(model.epoch_marker.item()) == 2
    assert metrics["selection_metric"] == "f1_score"
    assert metrics["best_epoch"] == 2
    assert metrics["best_selection_score"] == pytest.approx(46.153846, rel=1e-5)
    assert fold_score == pytest.approx(metrics["f1_score"])


def test_train_fold_can_still_select_by_accuracy_when_configured(monkeypatch):
    _patch_epoch_preference_training(monkeypatch)

    _, fold_score, model, metrics = fitness.train_fold_in_thread(
        {"optimizer": "adam", "learning_rate": 0.001},
        1,
        _selection_config("accuracy"),
        torch.device("cpu"),
    )

    assert int(model.epoch_marker.item()) == 1
    assert metrics["selection_metric"] == "accuracy"
    assert metrics["best_epoch"] == 1
    assert metrics["best_selection_score"] == pytest.approx(70.0)
    assert metrics["f1_score"] == pytest.approx(0.0)
    assert fold_score == pytest.approx(0.0)


def test_final_evaluation_records_selection_metric_metadata(monkeypatch):
    monkeypatch.setattr(cross_validation, "EvolvableCNN", EpochPreferenceModel)
    monkeypatch.setitem(cross_validation.OPTIMIZERS, "adam", NoOpOptimizer)

    result = cross_validation.evaluate_single_fold(
        {"optimizer": "adam", "learning_rate": 0.001},
        _selection_config("f1_score"),
        _one_batch_loader([0]),
        _one_batch_loader([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
        _one_batch_loader([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
        1,
        torch.device("cpu"),
        num_epochs=2,
    )

    assert result["selection_metric"] == "f1_score"
    assert result["best_epoch"] == 2
    assert result["best_selection_score"] == pytest.approx(46.153846, rel=1e-5)
    assert result["best_validation_acc"] == pytest.approx(30.0)
