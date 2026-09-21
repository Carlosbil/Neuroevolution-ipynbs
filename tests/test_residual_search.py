import copy
import random

import torch

from neuroevolution.config import get_default_config
from neuroevolution.genetics.crossover import crossover_genomes
from neuroevolution.genetics.innovation import build_innovation_genes
from neuroevolution.genetics.mutation import mutate_genome
from neuroevolution.models.evolvable_cnn import EvolvableCNN
from neuroevolution.models.genome_validator import (
    calculate_max_safe_conv_layers,
    is_genome_valid,
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
            "max_filters": 8,
            "min_fc_nodes": 4,
            "max_fc_nodes": 16,
            "kernel_size_options": [3, 5],
            "learning_rate_options": [0.001],
            "current_mutation_rate": 0.0,
            "crossover_rate": 1.0,
        }
    )
    config.update(overrides)
    return config


def base_genome(**overrides):
    genome = {
        "num_conv_layers": 2,
        "num_fc_layers": 1,
        "filters": [4, 4],
        "kernel_sizes": [3, 3],
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


def test_legacy_genome_gets_residual_defaults_and_remains_valid():
    config = small_config()
    legacy = base_genome()
    legacy.pop("residual_enabled", None)
    legacy.pop("residual_block_size", None)
    legacy.pop("residual_projection", None)

    fixed = validate_and_fix_genome(copy.deepcopy(legacy), config)

    assert fixed["residual_enabled"] is False
    assert fixed["residual_block_size"] == 2
    assert fixed["residual_projection"] == "auto"
    assert is_genome_valid(fixed, config) is True


def test_residual_depth_uses_block_level_pooling():
    config = small_config(sequence_length=32, max_conv_layers=8)
    sequential = base_genome(
        num_conv_layers=4,
        filters=[4, 4, 4, 4],
        kernel_sizes=[3, 3, 3, 3],
        residual_enabled=False,
    )
    residual = base_genome(
        num_conv_layers=4,
        filters=[4, 4, 4, 4],
        kernel_sizes=[3, 3, 3, 3],
        residual_enabled=True,
        residual_block_size=2,
        residual_projection="auto",
    )

    assert is_genome_valid(sequential, config) is False
    assert is_genome_valid(residual, config) is True
    assert calculate_max_safe_conv_layers(32) == 3
    assert calculate_max_safe_conv_layers(32, residual_enabled=True, residual_block_size=2) == 6


def test_residual_model_forward_uses_identity_shortcut_when_channels_match():
    config = small_config(num_channels=4)
    genome = base_genome(
        residual_enabled=True,
        residual_block_size=2,
        residual_projection="auto",
    )

    model = EvolvableCNN(genome, config)
    model.eval()
    residual_blocks = [
        layer for layer in model.conv_layers if layer.__class__.__name__ == "ResidualConv1DBlock"
    ]

    assert len(residual_blocks) == 1
    assert residual_blocks[0].shortcut.__class__.__name__ == "Identity"
    assert model(torch.randn(2, 4, 32)).shape == (2, 2)


def test_residual_model_forward_uses_projection_when_channels_differ():
    config = small_config(num_channels=1)
    genome = base_genome(
        residual_enabled=True,
        residual_block_size=2,
        residual_projection="auto",
    )

    model = EvolvableCNN(genome, config)
    model.eval()
    residual_blocks = [
        layer for layer in model.conv_layers if layer.__class__.__name__ == "ResidualConv1DBlock"
    ]

    assert len(residual_blocks) == 1
    assert residual_blocks[0].shortcut.__class__.__name__ == "Conv1d"
    assert model(torch.randn(2, 1, 32)).shape == (2, 2)


def test_mutation_and_crossover_keep_complete_residual_topology():
    random.seed(7)
    config = small_config(residual_mutation_weight=1.0)
    parent1 = validate_and_fix_genome(
        base_genome(
            residual_enabled=False,
            residual_block_size=2,
            residual_projection="auto",
            fitness=1.0,
            id="p1",
        ),
        config,
    )
    parent2 = validate_and_fix_genome(
        base_genome(
            filters=[6, 6],
            residual_enabled=True,
            residual_block_size=3,
            residual_projection="auto",
            fitness=0.5,
            id="p2",
        ),
        config,
    )
    parent1["innovation_genes"] = build_innovation_genes(parent1)
    parent2["innovation_genes"] = build_innovation_genes(parent2)

    mutated = mutate_genome(parent1, config)
    child1, child2 = crossover_genomes(parent1, parent2, config)

    for genome in (mutated, child1, child2):
        assert isinstance(genome["residual_enabled"], bool)
        assert genome["residual_block_size"] in config["residual_block_size_options"]
        assert genome["residual_projection"] in config["residual_projection_options"]
        assert is_genome_valid(genome, config)
        residual_gene_keys = {gene["gene_key"] for gene in genome["innovation_genes"]}
        assert {"residual_enabled", "residual_block_size", "residual_projection"} <= residual_gene_keys

    assert any(
        event["event_type"].startswith("mutate_residual")
        for event in mutated.get("structural_history", [])
    )
