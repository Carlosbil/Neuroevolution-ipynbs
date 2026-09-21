import random

import pytest
import torch

from neuroevolution.config import get_default_config, validate_config
from neuroevolution.evolution import engine
from neuroevolution.genetics.selection import (
    assign_pareto_metadata,
    dominates,
    non_dominated_sort,
    pareto_selection_key,
    tournament_selection,
)


OBJECTIVES = [
    {"name": "fitness", "direction": "maximize"},
    {"name": "evaluation_time_seconds", "direction": "minimize"},
    {"name": "parameter_count", "direction": "minimize"},
]


def _metrics():
    return {
        "accuracy": 10.0,
        "accuracy_std": 0.0,
        "sensitivity": 20.0,
        "sensitivity_std": 0.0,
        "specificity": 30.0,
        "specificity_std": 0.0,
        "precision": 40.0,
        "precision_std": 0.0,
        "f1_score": 50.0,
        "f1_score_std": 0.0,
        "auc": 60.0,
        "auc_std": 0.0,
        "fold_metrics": {},
        "n_valid_folds": 1,
        "fitness_split": "validation",
    }


def test_default_config_enables_cost_aware_pareto_selection():
    config = get_default_config()

    assert config["selection_strategy"] == "pareto"
    assert config["pareto_objectives"] == OBJECTIVES
    assert config["pareto_fitness_epsilon"] == 0.0

    validate_config(config)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"selection_strategy": "ranked"}, "selection_strategy"),
        ({"pareto_fitness_epsilon": -0.1}, "pareto_fitness_epsilon"),
        ({"pareto_objectives": [{"name": "unknown", "direction": "maximize"}]}, "objective"),
        ({"pareto_objectives": [{"name": "fitness", "direction": "higher"}]}, "direction"),
    ],
)
def test_config_validation_rejects_invalid_pareto_settings(override, message):
    config = get_default_config()
    config.update(override)

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_dominates_handles_maximize_minimize_objectives():
    cheaper = {
        "fitness": 91.0,
        "evaluation_time_seconds": 3.0,
        "parameter_count": 1000,
    }
    expensive = {
        "fitness": 91.0,
        "evaluation_time_seconds": 8.0,
        "parameter_count": 2000,
    }

    assert dominates(cheaper, expensive, OBJECTIVES)
    assert not dominates(expensive, cheaper, OBJECTIVES)


def test_fitness_epsilon_keeps_tiny_gain_from_dominating_cheaper_candidate():
    tiny_gain_expensive = {
        "fitness": 91.005,
        "evaluation_time_seconds": 10.0,
        "parameter_count": 2000,
    }
    cheaper = {
        "fitness": 91.0,
        "evaluation_time_seconds": 3.0,
        "parameter_count": 1000,
    }

    assert not dominates(tiny_gain_expensive, cheaper, OBJECTIVES, epsilon=0.01)


def test_non_dominated_sort_assigns_expected_fronts():
    population = [
        {"id": "balanced", "fitness": 91.0, "evaluation_time_seconds": 3.0, "parameter_count": 1000},
        {"id": "expensive", "fitness": 91.0, "evaluation_time_seconds": 8.0, "parameter_count": 2000},
        {"id": "fast", "fitness": 89.0, "evaluation_time_seconds": 1.0, "parameter_count": 700},
        {"id": "weak", "fitness": 80.0, "evaluation_time_seconds": 9.0, "parameter_count": 3000},
    ]

    fronts = non_dominated_sort(population, OBJECTIVES)

    assert [[individual["id"] for individual in front] for front in fronts] == [
        ["balanced", "fast"],
        ["expensive"],
        ["weak"],
    ]


def test_crowding_distance_prioritizes_boundary_points():
    population = [
        {"id": "fast", "fitness": 88.0, "evaluation_time_seconds": 1.0, "parameter_count": 900},
        {"id": "middle", "fitness": 90.0, "evaluation_time_seconds": 3.0, "parameter_count": 1100},
        {"id": "accurate", "fitness": 92.0, "evaluation_time_seconds": 5.0, "parameter_count": 1300},
    ]

    assign_pareto_metadata(population, OBJECTIVES)

    by_id = {individual["id"]: individual for individual in population}
    assert by_id["fast"]["crowding_distance"] == float("inf")
    assert by_id["accurate"]["crowding_distance"] == float("inf")
    assert by_id["middle"]["crowding_distance"] < float("inf")


def test_pareto_tournament_prefers_non_dominated_cheaper_candidate(monkeypatch):
    dominated_expensive = {
        "id": "expensive",
        "fitness": 91.0,
        "evaluation_time_seconds": 8.0,
        "parameter_count": 2000,
    }
    cheaper = {
        "id": "cheaper",
        "fitness": 91.0,
        "evaluation_time_seconds": 3.0,
        "parameter_count": 1000,
    }
    population = [dominated_expensive, cheaper]
    assign_pareto_metadata(population, OBJECTIVES)
    monkeypatch.setattr(random, "sample", lambda items, size: list(items))

    selected = tournament_selection(
        population,
        {
            "selection_strategy": "pareto",
            "pareto_objectives": OBJECTIVES,
            "pareto_fitness_epsilon": 0.0,
        },
        tournament_size=2,
    )

    assert selected["id"] == "cheaper"


def test_fitness_strategy_preserves_max_fitness_tournament(monkeypatch):
    population = [
        {"id": "cheap", "fitness": 89.0, "evaluation_time_seconds": 1.0, "parameter_count": 100},
        {"id": "fit", "fitness": 91.0, "evaluation_time_seconds": 9.0, "parameter_count": 2000},
    ]
    assign_pareto_metadata(population, OBJECTIVES)
    monkeypatch.setattr(random, "sample", lambda items, size: list(items))

    selected = tournament_selection(
        population,
        {"selection_strategy": "fitness"},
        tournament_size=2,
    )

    assert selected["id"] == "fit"


def test_pareto_selection_key_orders_rank_then_crowding_then_fitness():
    first_front_crowded = {"id": "a", "pareto_rank": 0, "crowding_distance": 0.1, "fitness": 90.0}
    first_front_diverse = {"id": "b", "pareto_rank": 0, "crowding_distance": 2.0, "fitness": 88.0}
    second_front_fit = {"id": "c", "pareto_rank": 1, "crowding_distance": float("inf"), "fitness": 99.0}

    ordered = sorted(
        [second_front_fit, first_front_crowded, first_front_diverse],
        key=lambda individual: pareto_selection_key(individual, {"selection_strategy": "pareto"}),
    )

    assert [individual["id"] for individual in ordered] == ["b", "a", "c"]


def test_evaluate_population_records_cost_metadata(monkeypatch, tmp_path):
    config = get_default_config(info_path=str(tmp_path))
    config.update(
        {
            "population_size": 2,
            "selection_strategy": "pareto",
            "num_folds": 1,
            "fold_parallel_workers": 1,
        }
    )
    neuroevolution = engine.HybridNeuroevolution(config, torch.device("cpu"))
    neuroevolution.population = [
        {"id": "g1", "optimizer": "adam", "learning_rate": 0.001},
        {"id": "g2", "optimizer": "adam", "learning_rate": 0.001},
    ]

    fitness_values = iter([90.0, 89.0])

    def fake_evaluate_fitness(genome, config, device):
        metrics = _metrics()
        metrics["f1_score"] = next(fitness_values)
        return metrics["f1_score"], object(), metrics

    counter = iter([1200, 900])
    monkeypatch.setattr(engine, "evaluate_fitness", fake_evaluate_fitness)
    monkeypatch.setattr(engine, "estimate_genome_parameter_count", lambda genome, config: next(counter))
    monkeypatch.setattr(engine.time, "perf_counter", iter([10.0, 10.5, 20.0, 21.25]).__next__)
    monkeypatch.setattr(
        engine.HybridNeuroevolution,
        "_format_architecture",
        lambda self, genome: "toy",
    )
    monkeypatch.setattr(engine.HybridNeuroevolution, "save_best_checkpoint", lambda *args, **kwargs: None)

    neuroevolution.evaluate_population()

    first, second = neuroevolution.population
    assert first["evaluation_time_seconds"] == pytest.approx(0.5)
    assert second["evaluation_time_seconds"] == pytest.approx(1.25)
    assert first["parameter_count"] == 1200
    assert second["parameter_count"] == 900
    assert "pareto_rank" in first
    assert "crowding_distance" in first
    assert first["metrics"]["evaluation_time_seconds"] == pytest.approx(0.5)
    assert first["metrics"]["parameter_count"] == 1200
    assert first["metrics"]["selection_objectives"]["fitness"] == pytest.approx(90.0)
    assert neuroevolution.generation_stats[-1]["pareto_objectives"] == OBJECTIVES
