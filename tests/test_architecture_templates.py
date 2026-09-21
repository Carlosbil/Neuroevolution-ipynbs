import copy
import random

import pytest
import torch

from neuroevolution.config import get_default_config, validate_config
from neuroevolution.evaluation.cross_validation import _format_architecture as format_cv_architecture
from neuroevolution.evolution.engine import HybridNeuroevolution
from neuroevolution.genetics.architecture_templates import (
    create_template_genome,
    get_template_registry,
    get_configured_templates,
)
from neuroevolution.genetics.innovation import build_innovation_genes
from neuroevolution.genetics.mutation import mutate_genome
from neuroevolution.models.evolvable_cnn import EvolvableCNN
from neuroevolution.models.genome_validator import (
    is_genome_valid,
    stable_genome_signature,
    validate_and_fix_genome,
)
from neuroevolution.visualization.reports import _format_architecture as format_report_architecture


EXPANDED_DEFAULT_TEMPLATE_IDS = [
    "lenet_conv1d_tiny",
    "alexnet_conv1d_small",
    "vgg_conv1d_small",
    "resnet_conv1d_small",
    "resnet_conv1d_medium",
    "wide_resnet_conv1d_small",
    "googlenet_inception_conv1d_small",
    "googlenet_inception_conv1d_medium",
    "inception_conv1d_wide",
]


def small_config(**overrides):
    config = get_default_config()
    config.update(
        {
            "population_size": 6,
            "sequence_length": 64,
            "num_channels": 1,
            "num_classes": 2,
            "min_conv_layers": 1,
            "max_conv_layers": 8,
            "initial_max_conv_layers": 4,
            "min_fc_layers": 1,
            "max_fc_layers": 3,
            "initial_max_fc_layers": 2,
            "min_filters": 2,
            "max_filters": 16,
            "min_fc_nodes": 4,
            "max_fc_nodes": 16,
            "kernel_size_options": [3, 5, 7],
            "learning_rate_options": [0.001],
            "current_mutation_rate": 0.0,
            "residual_mutation_weight": 0.0,
            "inception_mutation_weight": 0.0,
            "architecture_template_seed_fraction": 0.0,
            "architecture_template_mutation_weight": 0.0,
            "architecture_template_seed_min_random_fraction": 0.5,
            "architecture_template_ids": list(EXPANDED_DEFAULT_TEMPLATE_IDS),
            "architecture_template_max_attempts": 5,
            "crossover_rate": 1.0,
        }
    )
    config.update(overrides)
    return config


def base_genome(**overrides):
    genome = {
        "num_conv_layers": 3,
        "num_fc_layers": 1,
        "filters": [4, 6, 8],
        "kernel_sizes": [3, 5, 7],
        "fc_nodes": [8],
        "activations": ["relu", "relu", "relu"],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "normalization_type": "batch",
        "conv_topology": "sequential",
        "residual_enabled": False,
        "residual_block_size": 2,
        "residual_projection": "auto",
        "inception_enabled": False,
        "inception_reduction_ratio": 0.5,
        "inception_pool_branch": True,
        "fitness": 0.0,
        "id": "base",
        "structural_history": [],
    }
    genome.update(overrides)
    return genome


def test_template_registry_filters_configured_ids_and_rejects_unknown_templates():
    registry = get_template_registry()

    assert set(EXPANDED_DEFAULT_TEMPLATE_IDS) <= set(registry)
    assert registry["resnet_conv1d_small"].template_family == "resnet"
    assert registry["googlenet_inception_conv1d_small"].conv_topology == "inception"

    default_config = get_default_config()
    assert default_config["architecture_template_ids"] == EXPANDED_DEFAULT_TEMPLATE_IDS

    config = small_config(architecture_template_ids=["resnet_conv1d_small"])
    configured = get_configured_templates(config)

    assert [template.template_id for template in configured] == ["resnet_conv1d_small"]

    invalid = small_config(architecture_template_ids=["missing_template"])
    with pytest.raises(ValueError, match="unknown architecture template"):
        validate_config(invalid)


def test_all_default_templates_build_valid_models():
    config = small_config()
    registry = get_template_registry()

    observed_topologies = set()
    for template_id in EXPANDED_DEFAULT_TEMPLATE_IDS:
        template = registry[template_id]
        genome = create_template_genome(template_id, config, origin="initial_seed")
        observed_topologies.add(genome["conv_topology"])

        assert genome["architecture_template_id"] == template_id
        assert genome["architecture_template_family"] == template.template_family
        assert is_genome_valid(genome, config)
        assert genome["innovation_genes"]

        model = EvolvableCNN(copy.deepcopy(genome), config)
        model.eval()
        assert model(torch.randn(2, 1, 64)).shape == (2, 2)

    assert observed_topologies == {"sequential", "residual", "inception"}


def test_template_factory_returns_valid_normalized_genome_with_provenance():
    config = small_config()

    resnet = create_template_genome("resnet_conv1d_small", config, origin="initial_seed")
    googlenet = create_template_genome(
        "googlenet_inception_conv1d_small",
        config,
        origin="initial_seed",
    )

    for genome, expected_family in ((resnet, "resnet"), (googlenet, "googlenet")):
        assert genome["architecture_template_family"] == expected_family
        assert genome["architecture_template_origin"] == "initial_seed"
        assert is_genome_valid(genome, config)
        assert genome["innovation_genes"]
        assert any(
            event["event_type"] == "architecture_template_seed"
            for event in genome["structural_history"]
        )
        model = EvolvableCNN(copy.deepcopy(genome), config)
        model.eval()
        assert model(torch.randn(2, 1, 64)).shape == (2, 2)

    assert resnet["conv_topology"] == "residual"
    assert resnet["residual_enabled"] is True
    assert googlenet["conv_topology"] == "inception"
    assert googlenet["inception_enabled"] is True


def test_initial_population_template_quota_preserves_random_slots_and_double_cap(tmp_path):
    random.seed(13)
    config = small_config(
        artifacts_dir=str(tmp_path),
        population_size=6,
        architecture_template_seed_fraction=1.0,
        architecture_template_seed_min_random_fraction=0.5,
    )
    engine = HybridNeuroevolution(config, torch.device("cpu"))

    assert engine._template_seed_quota(5) == 2

    engine.initialize_population()

    assert len(engine.population) == 6
    template_seeds = [
        genome for genome in engine.population
        if genome.get("architecture_template_origin") == "initial_seed"
    ]
    random_seeds = [
        genome for genome in engine.population
        if genome.get("architecture_template_origin") in (None, "random")
    ]

    assert len(template_seeds) == 2
    assert len(random_seeds) >= 3
    assert any(
        event["event_type"] == "double_cap_seed"
        for genome in engine.population
        for event in genome.get("structural_history", [])
    )
    assert all(genome["num_conv_layers"] <= config["max_conv_layers"] for genome in template_seeds)
    assert all(
        any(event["event_type"] == "architecture_template_seed" for event in genome["structural_history"])
        for genome in template_seeds
    )


def test_template_seeding_disabled_keeps_population_random(tmp_path):
    random.seed(17)
    config = small_config(artifacts_dir=str(tmp_path), architecture_template_seed_fraction=0.0)
    engine = HybridNeuroevolution(config, torch.device("cpu"))

    engine.initialize_population()

    assert not any(
        genome.get("architecture_template_origin") == "initial_seed"
        for genome in engine.population
    )
    assert any(
        event["event_type"] == "quartile_init"
        for genome in engine.population
        for event in genome.get("structural_history", [])
    )


def test_template_module_mutation_returns_valid_evolvable_genome_with_event():
    random.seed(23)
    config = small_config(
        architecture_template_mutation_weight=1.0,
        architecture_template_ids=["googlenet_inception_conv1d_small"],
    )
    genome = validate_and_fix_genome(base_genome(skip_next_evaluation=True), config)
    genome["innovation_genes"] = build_innovation_genes(genome)

    mutated = mutate_genome(genome, config)

    assert mutated["id"] != genome["id"]
    assert "skip_next_evaluation" not in mutated
    assert mutated["architecture_template_id"] == "googlenet_inception_conv1d_small"
    assert mutated["architecture_template_origin"] == "mutation_module"
    assert mutated["conv_topology"] == "inception"
    assert is_genome_valid(mutated, config)
    assert mutated["innovation_genes"]
    assert any(
        event["event_type"] == "architecture_template_module"
        for event in mutated["structural_history"]
    )


def test_zero_template_mutation_weight_keeps_standard_mutation_available():
    random.seed(29)
    config = small_config(current_mutation_rate=1.0, architecture_template_mutation_weight=0.0)
    genome = validate_and_fix_genome(base_genome(), config)

    mutated = mutate_genome(genome, config)

    assert mutated.get("architecture_template_origin") in (None, "random")
    assert any(
        event["event_type"].startswith("mutate_")
        for event in mutated.get("structural_history", [])
    )


def test_template_provenance_is_visible_without_affecting_cache_signature():
    config = small_config()
    plain = validate_and_fix_genome(
        base_genome(
            conv_topology="residual",
            residual_enabled=True,
            residual_block_size=2,
            residual_projection="auto",
        ),
        config,
    )
    templated = copy.deepcopy(plain)
    templated.update(
        {
            "architecture_template_id": "resnet_conv1d_small",
            "architecture_template_family": "resnet",
            "architecture_template_origin": "initial_seed",
        }
    )

    assert stable_genome_signature(plain, config) == stable_genome_signature(templated, config)
    assert "tpl=resnet_conv1d_small" in format_cv_architecture(templated)
    assert "tpl=resnet_conv1d_small" in format_report_architecture(templated)

    model = EvolvableCNN(copy.deepcopy(templated), config)
    summary = model.get_architecture_summary()

    assert "Template: resnet_conv1d_small" in summary
    assert "Template Family: resnet" in summary
