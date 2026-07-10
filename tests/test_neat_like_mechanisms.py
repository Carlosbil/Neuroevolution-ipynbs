import copy
import random

import torch

from neuroevolution.evolution.engine import HybridNeuroevolution
from neuroevolution.genetics.genome import create_random_genome
from neuroevolution.genetics.innovation import (
    append_structural_event,
    build_innovation_genes,
    innovation_uuid,
)
from neuroevolution.genetics import crossover as crossover_module
from neuroevolution.genetics.speciation import (
    assign_species,
    calculate_compatibility_distance,
    calculate_species_adjusted_fitness,
    update_species_representatives,
)


def _config(tmp_path=None):
    artifacts_dir = str(tmp_path) if tmp_path is not None else "artifacts/test_neat_like"
    return {
        "artifacts_dir": artifacts_dir,
        "population_size": 4,
        "max_generations": 5,
        "fitness_threshold": 99.0,
        "base_mutation_rate": 0.25,
        "mutation_rate_min": 0.10,
        "mutation_rate_max": 0.80,
        "current_mutation_rate": 0.25,
        "structural_mutation_generation_factor": 0.5,
        "crossover_rate": 1.0,
        "elite_percentage": 0.25,
        "early_stopping_generations": 3,
        "min_improvement_threshold": 0.01,
        "min_conv_layers": 1,
        "max_conv_layers": 4,
        "min_fc_layers": 1,
        "max_fc_layers": 3,
        "min_filters": 1,
        "max_filters": 128,
        "min_fc_nodes": 4,
        "max_fc_nodes": 256,
        "kernel_size_options": [1, 3, 5],
        "min_dropout": 0.1,
        "max_dropout": 0.5,
        "learning_rate_options": [0.001, 0.0001],
        "normalization_batch_weight": 0.8,
        "normalization_layer_weight": 0.2,
        "sequence_length": 64,
        "complexity_step_generations": 2,
        "initial_max_conv_layers": 1,
        "initial_max_fc_layers": 1,
        "speciation_threshold": 0.20,
    }


def _genome(
    genome_id,
    filters,
    kernel_sizes,
    fc_nodes,
    *,
    fitness=0.0,
    dropout_rate=0.2,
    learning_rate=0.001,
):
    genome = {
        "id": genome_id,
        "num_conv_layers": len(filters),
        "num_fc_layers": len(fc_nodes),
        "filters": list(filters),
        "kernel_sizes": list(kernel_sizes),
        "fc_nodes": list(fc_nodes),
        "activations": ["relu"] * max(len(filters), len(fc_nodes)),
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "optimizer": "adam",
        "normalization_type": "batch",
        "fitness": fitness,
        "structural_history": [],
    }
    genome["innovation_genes"] = build_innovation_genes(genome)
    return genome


def test_innovation_uuid_is_deterministic_and_payload_order_independent():
    first = innovation_uuid("conv_filter", {"index": 0, "value": 16})
    second = innovation_uuid("conv_filter", {"value": 16, "index": 0})
    different = innovation_uuid("conv_filter", {"index": 0, "value": 32})

    assert first == second
    assert first != different


def test_build_innovation_genes_and_structural_history_track_events():
    genome = _genome("g1", [16, 32], [3, 5], [64])

    gene_keys = {gene["gene_key"] for gene in genome["innovation_genes"]}
    assert gene_keys == {
        "conv_filter_0",
        "conv_filter_1",
        "conv_kernel_0",
        "conv_kernel_1",
        "fc_node_0",
    }
    assert all(gene["enabled"] for gene in genome["innovation_genes"])

    append_structural_event(genome, "mutate_conv_filter", {"index": 1, "old": 16, "new": 32})

    assert genome["structural_history"] == [
        {
            "innovation_id": innovation_uuid(
                "mutate_conv_filter",
                {"index": 1, "old": 16, "new": 32},
            ),
            "event_type": "mutate_conv_filter",
            "payload": {"index": 1, "old": 16, "new": 32},
        }
    ]


def test_crossover_aligns_homologous_innovation_genes(monkeypatch):
    config = _config()
    parent1 = _genome("p1", [16], [3], [64], fitness=10.0)
    parent2 = _genome("p2", [16, 32], [3, 5], [64, 128], fitness=1.0)

    monkeypatch.setattr(crossover_module.random, "random", lambda: 0.0)

    child1, child2 = crossover_module.crossover_genomes(parent1, parent2, config)

    assert child1["id"] not in {"p1", "p2"}
    assert child1["fitness"] == 0.0
    assert child1["filters"] == [16, 32]
    assert child1["kernel_sizes"] == [3, 5]
    assert child1["fc_nodes"] == [64, 128]
    assert child1["num_conv_layers"] == 2
    assert child1["num_fc_layers"] == 2
    assert child1["structural_history"][-1]["event_type"] == "innovation_crossover"
    assert child1["structural_history"][-1]["payload"]["dominant_parent"] == "p1"
    assert child1["innovation_genes"] == build_innovation_genes(child1)

    assert child2["id"] not in {"p1", "p2"}
    assert child2["innovation_genes"] == build_innovation_genes(child2)


def test_assign_species_groups_close_genomes_and_computes_adjusted_fitness():
    config = _config()
    close_a = _genome("close-a", [16], [3], [64], fitness=6.0)
    close_b = copy.deepcopy(close_a)
    close_b["id"] = "close-b"
    close_b["fitness"] = 4.0
    distant = _genome(
        "distant",
        [16, 32],
        [3, 5],
        [64, 128],
        fitness=8.0,
        dropout_rate=0.5,
        learning_rate=0.0001,
    )

    species = assign_species([close_a, close_b, distant], {}, config)

    species_sizes = sorted(len(specie["members"]) for specie in species.values())
    assert species_sizes == [1, 2]
    assert close_a["species_id"] == close_b["species_id"]
    assert distant["species_id"] != close_a["species_id"]
    assert calculate_compatibility_distance(close_a, close_b, config) == 0.0
    assert calculate_compatibility_distance(close_a, distant, config) > config["speciation_threshold"]

    calculate_species_adjusted_fitness(species)
    paired_species = next(specie for specie in species.values() if len(specie["members"]) == 2)
    adjusted_by_id = {genome["id"]: genome["adjusted_fitness"] for genome in paired_species["members"]}
    assert adjusted_by_id == {"close-a": 3.0, "close-b": 2.0}

    update_species_representatives(species)
    assert paired_species["representative"]["id"] == "close-a"


def test_engine_speciation_assigns_species_ids(tmp_path):
    config = _config(tmp_path)
    engine = HybridNeuroevolution(config=config, device=torch.device("cpu"))
    close_a = _genome("close-a", [16], [3], [64])
    close_b = copy.deepcopy(close_a)
    close_b["id"] = "close-b"
    distant = _genome("distant", [16, 32], [3, 5], [64, 128])

    engine.population = [close_a, close_b, distant]
    engine._speciate_population()

    assert sorted(len(specie["members"]) for specie in engine.species.values()) == [1, 2]
    assert close_a["species_id"] == close_b["species_id"]
    assert distant["species_id"] != close_a["species_id"]


def test_incremental_complexity_caps_advance_by_generation_stage(tmp_path):
    config = _config(tmp_path)
    engine = HybridNeuroevolution(config=config, device=torch.device("cpu"))

    engine.generation = 0
    engine._update_incremental_complexity()
    assert engine.config["current_max_conv_layers"] == 1
    assert engine.config["current_max_fc_layers"] == 1

    engine.generation = 2
    engine._update_incremental_complexity()
    assert engine.config["current_max_conv_layers"] == 2
    assert engine.config["current_max_fc_layers"] == 2

    engine.generation = 99
    engine._update_incremental_complexity()
    assert engine.config["current_max_conv_layers"] == config["max_conv_layers"]
    assert engine.config["current_max_fc_layers"] == config["max_fc_layers"]


def test_random_genome_creation_respects_active_incremental_caps():
    config = _config()
    config["current_max_conv_layers"] = 2
    config["current_max_fc_layers"] = 1
    random.seed(123)

    for _ in range(20):
        genome = create_random_genome(config)
        assert 1 <= genome["num_conv_layers"] <= 2
        assert genome["num_fc_layers"] == 1
        assert len(genome["filters"]) == genome["num_conv_layers"]
        assert len(genome["kernel_sizes"]) == genome["num_conv_layers"]
        assert len(genome["fc_nodes"]) == genome["num_fc_layers"]
        assert genome["innovation_genes"] == build_innovation_genes(genome)
