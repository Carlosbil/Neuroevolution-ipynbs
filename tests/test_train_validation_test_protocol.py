import numpy as np
import torch

from neuroevolution.evolution import fitness as fitness_module
from neuroevolution.evolution.fitness import (
    checkpoint_selection_score,
    load_fold_loaders,
    train_fold_in_thread,
)
from neuroevolution.evolution.engine import HybridNeuroevolution
from neuroevolution.evaluation import cross_validation
from neuroevolution.config import get_default_config


def _write_fold_arrays(base_dir, dataset_id="demo", fold=1):
    fold_dir = base_dir / "folds"
    fold_dir.mkdir()

    split_values = {
        "train": (np.array([[1.0, 1.1], [2.0, 2.1]], dtype=np.float32), np.array([0, 1])),
        "val": (np.array([[10.0, 10.1], [11.0, 11.1], [12.0, 12.1]], dtype=np.float32), np.array([1, 0, 1])),
        "test": (np.array([[20.0, 20.1]], dtype=np.float32), np.array([0])),
    }

    for split, (features, labels) in split_values.items():
        np.save(fold_dir / f"X_{split}_{dataset_id}_fold_{fold}.npy", features)
        np.save(fold_dir / f"y_{split}_{dataset_id}_fold_{fold}.npy", labels)

    return fold_dir


def _config(tmp_path):
    return {
        "data_path": str(tmp_path),
        "fold_files_subdirectory": "folds",
        "dataset_id": "demo",
        "fold_id": "demo",
        "batch_size": 2,
        "dataloader_num_workers": 0,
        "dataloader_persistent_workers": False,
        "dataloader_pin_memory": False,
        "dataloader_prefetch_factor": 1,
        "fold_cache_mode": "none",
    }


def _all_targets(loader):
    targets = []
    for _, batch_targets in loader:
        targets.extend(batch_targets.tolist())
    return targets


def _loader(features, labels, batch_size=2):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(features, dtype=torch.float32).reshape(len(labels), 1, 1),
        torch.tensor(labels, dtype=torch.long),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


class SignModel(torch.nn.Module):
    def __init__(self, genome, config):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        score = self.scale * data[:, 0, 0]
        return torch.stack([-score, score], dim=1)


def test_load_fold_loaders_keeps_validation_and_test_separate(tmp_path):
    _write_fold_arrays(tmp_path)

    loaders = load_fold_loaders(1, _config(tmp_path), torch.device("cpu"))

    assert len(loaders.train.dataset) == 2
    assert len(loaders.validation.dataset) == 3
    assert len(loaders.test.dataset) == 1
    assert _all_targets(loaders.validation) == [1, 0, 1]
    assert _all_targets(loaders.test) == [0]


def test_final_evaluation_reports_held_out_test_metrics(monkeypatch):
    monkeypatch.setattr(cross_validation, "EvolvableCNN", SignModel)

    train_loader = _loader([[-1.0], [1.0]], [0, 1])
    validation_loader = _loader([[-1.0], [1.0], [1.0]], [0, 1, 1])
    test_loader = _loader([[1.0]], [1])

    result = cross_validation.evaluate_single_fold(
        best_genome={"optimizer": "sgd", "learning_rate": 0.0},
        config={"epoch_patience": 1, "improvement_threshold": 0.0},
        fold_train_loader=train_loader,
        fold_validation_loader=validation_loader,
        fold_test_loader=test_loader,
        fold_num=1,
        device=torch.device("cpu"),
        num_epochs=1,
    )

    assert result["selection_split"] == "validation"
    assert result["evaluation_split"] == "test"
    assert result["n_samples"] == 1
    assert result["accuracy"] == 100.0


def test_checkpoint_selection_score_defaults_to_f1_score():
    metrics = {"accuracy": 99.0, "f1_score": 25.0}

    assert checkpoint_selection_score(metrics, {}) == 25.0
    assert checkpoint_selection_score(metrics, {"checkpoint_metric": "accuracy"}) == 99.0


def test_default_config_aligns_checkpoint_and_fitness_metrics():
    config = get_default_config()

    assert config["fitness_metric"] == "f1_score"
    assert config["checkpoint_metric"] == "f1_score"


def test_evolution_fitness_uses_validation_not_test(tmp_path, monkeypatch):
    monkeypatch.setattr(fitness_module, "EvolvableCNN", SignModel)
    _write_fold_arrays(tmp_path)

    config = _config(tmp_path)
    config.update(
        {
            "num_epochs": 1,
            "validation_frequency_epochs": 1,
            "early_stopping_patience": 100,
            "epoch_patience": 1,
            "improvement_threshold": 0.0,
            "use_amp": False,
        }
    )

    # Validation predicts class 1 correctly; test would reduce F1 if combined.
    fold_dir = tmp_path / "folds"
    np.save(fold_dir / "X_val_demo_fold_1.npy", np.array([[1.0], [1.0]], dtype=np.float32))
    np.save(fold_dir / "y_val_demo_fold_1.npy", np.array([1, 1]))
    np.save(fold_dir / "X_test_demo_fold_1.npy", np.array([[1.0], [1.0]], dtype=np.float32))
    np.save(fold_dir / "y_test_demo_fold_1.npy", np.array([0, 0]))

    _, score, _, metrics = train_fold_in_thread(
        {"id": "g1", "optimizer": "sgd", "learning_rate": 0.0},
        fold_num=1,
        config=config,
        device=torch.device("cpu"),
    )

    assert score == 100.0
    assert metrics["f1_score"] == 100.0
    assert metrics["selection_split"] == "validation"


def test_global_checkpoint_records_validation_selection_metadata(tmp_path):
    config = {
        "artifacts_dir": str(tmp_path),
        "min_conv_layers": 1,
        "min_fc_layers": 1,
        "max_conv_layers": 1,
        "max_fc_layers": 1,
        "checkpoint_metric": "f1_score",
    }
    engine = HybridNeuroevolution(config=config, device=torch.device("cpu"))
    model = torch.nn.Linear(1, 1)

    engine.save_best_checkpoint({"id": "abc123", "fitness": 42.0}, model)

    checkpoint = torch.load(engine.best_checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["selection_split"] == "validation"
    assert checkpoint["fitness_metric"] == "f1_score"
    assert checkpoint["checkpoint_metric"] == "f1_score"
    assert checkpoint["test_evaluated"] is False
