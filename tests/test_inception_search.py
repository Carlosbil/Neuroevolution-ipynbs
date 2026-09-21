import copy
import random

import torch

from neuroevolution.config import get_default_config
from neuroevolution.evaluation.cross_validation import _format_architecture as format_cv_architecture
from neuroevolution.genetics.crossover import crossover_genomes
from neuroevolution.genetics.genome import create_random_genome
from neuroevolution.genetics.innovation import build_innovation_genes
from neuroevolution.genetics.mutation import mutate_genome
from neuroevolution.genetics.speciation import calculate_compatibility_distance
from neuroevolution.models.evolvable_cnn import EvolvableCNN
from neuroevolution.models.genome_validator import (
    calculate_inception_branch_channels,
    is_genome_valid,
    stable_genome_signature,
    validate_and_fix_genome,
)


def small_config(**overrides):
    config = get_default_config()
    config.update(
        {
            "population_size": 4,
            "sequence_length": 32,
            "num_channels": 1,
            "num_classes": 2,
            "min_conv_layers": 1,
            "max_conv_layers": 8,
            "min_fc_layers": 1,
            "max_fc_layers": 3,
            "min_filters": 2,
            "max_filters": 16,
            "min_fc_nodes": 4,
            "max_fc_nodes": 16,
            "kernel_size_options": [3, 5, 7],
            "learning_rate_options": [0.001],
            "current_mutation_rate": 0.0,
            "residual_mutation_weight": 0.0,
            "inception_mutation_weight": 0.0,
            "crossover_rate": 1.0,
            "conv_topology_weights": {
                "sequential": 1.0,
                "residual": 0.0,
                "inception": 0.0,
            },
        }
    )
    config.update(overrides)
    return config


def base_genome(**overrides):
    genome = {
        "num_conv_layers": 2,
        "num_fc_layers": 1,
        "filters": [8, 8],
        "kernel_sizes": [5, 7],
        "fc_nodes": [8],
        "activations": ["relu", "relu"],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "normalization_type": "batch",
        "fitness": 0.0,
        "id": "test",
        "structural_history": [],
    }
    genome.update(overrides)
    return genome


def test_legacy_genome_gets_inception_defaults_and_remains_valid():
    config = small_config()
    legacy = base_genome()
    for key in (
        "inception_enabled",
        "inception_reduction_ratio",
        "inception_pool_branch",
        "conv_topology",
    ):
        legacy.pop(key, None)

    fixed = validate_and_fix_genome(copy.deepcopy(legacy), config)

    assert fixed["inception_enabled"] is False
    assert fixed["inception_reduction_ratio"] == 0.5
    assert fixed["inception_pool_branch"] is True
    assert fixed["conv_topology"] == "sequential"
    assert is_genome_valid(fixed, config) is True


def test_inception_topology_is_mutually_exclusive_with_residual():
    config = small_config()
    conflicting = base_genome(
        conv_topology="inception",
        residual_enabled=True,
        residual_block_size=2,
        residual_projection="auto",
        inception_enabled=True,
        inception_reduction_ratio=0.25,
        inception_pool_branch=True,
    )

    fixed = validate_and_fix_genome(copy.deepcopy(conflicting), config)

    assert fixed["conv_topology"] == "inception"
    assert fixed["inception_enabled"] is True
    assert fixed["residual_enabled"] is False


def test_inception_branch_channels_are_split_and_validated():
    split = calculate_inception_branch_channels(
        total_channels=10,
        pool_branch=True,
        min_branch_channels=1,
    )

    assert set(split) == {"pointwise", "medium", "wide", "pool"}
    assert sum(split.values()) == 10
    assert min(split.values()) >= 1

    config = small_config(max_filters=7, inception_min_branch_channels=2)
    invalid = validate_and_fix_genome(
        base_genome(
            num_conv_layers=1,
            filters=[3],
            kernel_sizes=[3],
            conv_topology="inception",
            inception_enabled=True,
            inception_pool_branch=True,
        ),
        config,
    )

    assert invalid["filters"][0] == 3
    assert is_genome_valid(invalid, config) is False


def test_inception_model_forward_with_and_without_pool_branch():
    config = small_config()

    for pool_branch, expected_branches in ((True, 4), (False, 3)):
        genome = validate_and_fix_genome(
            base_genome(
                conv_topology="inception",
                inception_enabled=True,
                inception_reduction_ratio=0.5,
                inception_pool_branch=pool_branch,
            ),
            config,
        )
        model = EvolvableCNN(genome, config)
        model.eval()
        modules = [
            layer for layer in model.conv_layers
            if layer.__class__.__name__ == "InceptionConv1DModule"
        ]

        assert len(modules) == genome["num_conv_layers"]
        assert len(modules[0].branch_channels) == expected_branches
        assert model(torch.randn(2, 1, 32)).shape == (2, 2)


def test_creation_mutation_crossover_and_innovation_handle_inception_topology():
    random.seed(11)
    config = small_config(
        conv_topology_weights={"sequential": 0.0, "residual": 0.0, "inception": 1.0},
        inception_mutation_weight=1.0,
    )

    created = create_random_genome(config)
    assert created["conv_topology"] == "inception"
    assert created["inception_enabled"] is True
    assert created["residual_enabled"] is False

    sequential = validate_and_fix_genome(base_genome(id="p1", fitness=1.0), config)
    inception = validate_and_fix_genome(
        base_genome(
            id="p2",
            fitness=0.5,
            conv_topology="inception",
            residual_enabled=False,
            inception_enabled=True,
            inception_reduction_ratio=0.25,
            inception_pool_branch=False,
        ),
        config,
    )
    sequential["innovation_genes"] = build_innovation_genes(sequential)
    inception["innovation_genes"] = build_innovation_genes(inception)

    mutated = mutate_genome(sequential, config)
    child1, child2 = crossover_genomes(sequential, inception, config)

    for genome in (mutated, child1, child2):
        assert genome["conv_topology"] in {"sequential", "residual", "inception"}
        assert not (genome.get("residual_enabled", False) and genome.get("inception_enabled", False))
        assert genome["inception_reduction_ratio"] in config["inception_reduction_ratio_options"]
        assert genome["inception_pool_branch"] in config["inception_pool_branch_options"]
        assert is_genome_valid(genome, config)
        gene_keys = {gene["gene_key"] for gene in genome["innovation_genes"]}
        assert {
            "inception_enabled",
            "inception_reduction_ratio",
            "inception_pool_branch",
        } <= gene_keys

    assert any(
        event["event_type"].startswith("mutate_inception")
        for event in mutated.get("structural_history", [])
    )


def test_signatures_speciation_and_summaries_distinguish_inception():
    config = small_config()
    sequential = validate_and_fix_genome(base_genome(), config)
    inception = validate_and_fix_genome(
        base_genome(
            conv_topology="inception",
            inception_enabled=True,
            inception_reduction_ratio=0.25,
            inception_pool_branch=False,
        ),
        config,
    )
    sequential["innovation_genes"] = build_innovation_genes(sequential)
    inception["innovation_genes"] = build_innovation_genes(inception)

    assert stable_genome_signature(sequential, config) != stable_genome_signature(inception, config)
    assert calculate_compatibility_distance(sequential, inception, config) > 0

    model = EvolvableCNN(inception, config)
    summary = model.get_architecture_summary()

    assert "Topology: inception" in summary
    assert "Inception Reduction Ratio: 0.25" in summary
    assert "Inception Pool Branch: False" in summary
    assert format_cv_architecture(inception) == "incep, 2C, 1FC"
