"""Scheduled reference entry, distinct parents and changed policies on resume."""

import copy
import json
from pathlib import Path

import pytest
import torch

from neuroevolution.config import (
    get_default_config, get_operator_config,
    reference_injection_fraction_for_generation, validate_config,
)
from neuroevolution.evolution import engine as engine_module
from neuroevolution.evolution.engine import HybridNeuroevolution
from neuroevolution.genetics.architecture_templates import create_template_genome


def small_config(tmp_path, **overrides):
    config = get_default_config(str(tmp_path))
    config.update(
        sequence_length=64, pre_fc_pool_length=4,
        min_filters=4, max_filters=8, min_fc_nodes=4, max_fc_nodes=8,
        max_conv_layers=3, initial_max_conv_layers=1,
        max_fc_layers=1, architecture_template_ids=['lenet5', 'resnet18'],
    )
    config.update(overrides)
    return config


def score_population(evolution):
    for index, genome in enumerate(evolution.population):
        genome['fitness'] = 50 + index
        genome['metrics'] = {'f1_score': 50 + index}


def current_injections(evolution):
    return [genome for genome in evolution.population
            if any(event['event_type'] == 'architecture_template_seed'
                   and event['payload'].get('generation') == evolution.generation
                   for event in genome.get('structural_history', []))]


@pytest.mark.parametrize('generation, expected', [(0, 0), (9, 0), (10, 1), (19, 1), (20, 2), (99, 2)])
def test_stage_boundaries(generation, expected):
    config = get_default_config()
    assert int(config['population_size'] * reference_injection_fraction_for_generation(config, generation)) == expected


def test_population_remains_custom_through_warmup_then_injects_exact_quotas(tmp_path):
    # Even explicit legacy seeding/mutation settings cannot bypass the schedule.
    config = small_config(tmp_path, architecture_template_seed_fraction=1.0,
                          architecture_template_mutation_weight=1.0)
    before = copy.deepcopy(config)
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    evolution.initialize_population()
    assert not any(g.get('model_kind') == 'reference' for g in evolution.population)
    for generation in range(1, 22):
        score_population(evolution)
        evolution.generation = generation
        evolution._update_incremental_complexity()
        evolution.selection_and_reproduction()
        assert len(evolution.population) == config['population_size'] == 10
        assert len(current_injections(evolution)) == (0 if generation < 10 else 1 if generation < 20 else 2)
        if generation < 10:
            assert not any(g.get('model_kind') == 'reference' for g in evolution.population)
    assert config == before == evolution.config


@pytest.mark.parametrize('generation', [9, 19, 28])
def test_resume_uses_absolute_generation_preserves_elites_and_template_cursor(tmp_path, generation):
    config = small_config(tmp_path)
    previous = HybridNeuroevolution(config, torch.device('cpu'))
    previous.initialize_population()
    previous.generation = generation
    previous.template_cursor = 3
    if generation == 28:
        previous.population[0] = create_template_genome('lenet5', config)
        previous.population[0]['fitness'] = 100
    else:
        score_population(previous)
    previous._save_evolution_progress()
    before = Path(previous.progress_json_path).read_bytes()
    resumed = HybridNeuroevolution(config, torch.device('cpu'))
    resumed.initialize_population()
    assert Path(previous.progress_json_path).read_bytes() == before
    assert resumed.population == previous.population
    resumed.generation += 1
    resumed._update_incremental_complexity()
    resumed.selection_and_reproduction()
    injected = current_injections(resumed)
    expected = 1 if generation == 9 else 2
    assert len(injected) == expected
    assert resumed.template_cursor == 3 + expected
    assert injected[0]['reference_architecture_id'] == 'resnet18'
    if generation == 28:
        assert any(g['id'] == previous.population[0]['id'] and g['skip_next_evaluation']
                   for g in resumed.population)


def test_legacy_constant_injection_and_template_mutation_can_be_enabled(tmp_path):
    config = small_config(tmp_path, reference_injection_schedule=None,
                          reference_injection_fraction=0.10,
                          architecture_template_mutation_weight=0.8)
    assert reference_injection_fraction_for_generation(config, 1) == 0.10
    assert get_operator_config(config, 1, 0.15)['architecture_template_mutation_weight'] == 0.8


@pytest.mark.parametrize('schedule', [[], [(1, 0.0)], [(0, 0.0), (0, 0.1)],
                                     [(0, 0.0), (2.5, 0.1)], [(0, -0.1)],
                                     [(0, float('nan'))], [(0, float('inf'))],
                                     [(0, 1.1)], [(0,)], 'invalid'])
def test_invalid_schedule_is_rejected(schedule):
    config = get_default_config()
    config['reference_injection_schedule'] = schedule
    with pytest.raises(ValueError, match='reference_injection_schedule'):
        validate_config(config)


@pytest.mark.parametrize('species_size', [0, 1, 2])
def test_reproduction_uses_distinct_parents_even_with_singleton_survivors(tmp_path, monkeypatch, species_size):
    config = small_config(tmp_path, population_size=6, speciation_enabled=species_size > 0,
                          random_immigrant_fraction=0, species_elite_fraction=0)
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    evolution.initialize_population()
    score_population(evolution)
    if species_size:
        def speciate():
            evolution.species = {
                str(i): {'members': evolution.population[i:i + species_size]}
                for i in range(0, 6, species_size)
            }
        monkeypatch.setattr(evolution, '_speciate_population', speciate)
    # Deterministic selection reproduces the former self-crossover bug.
    monkeypatch.setattr(engine_module, 'select_tournament', lambda population, *args: population[0])
    original = engine_module.crossover_genomes
    pairs = []
    def record_crossover(first, second, operator_config):
        pairs.append((first['id'], second['id']))
        return original(first, second, operator_config)
    monkeypatch.setattr(engine_module, 'crossover_genomes', record_crossover)
    evolution.generation = 1
    evolution.selection_and_reproduction()
    assert pairs and all(first != second for first, second in pairs)


def test_resume_recomputes_mutation_when_search_policy_changes(tmp_path):
    old_config = small_config(tmp_path, base_mutation_rate=0.20,
                              current_mutation_rate=0.20,
                              mutation_rate_max=0.50, mutation_stagnation_boost=0.15)
    previous = HybridNeuroevolution(old_config, torch.device('cpu'))
    previous.initialize_population()
    previous.generation = 28
    previous.generations_without_improvement = 19
    previous.generation_stats = [{'std_fitness': 1}]
    previous.current_mutation_rate = 0.355
    previous._save_evolution_progress()
    resumed = HybridNeuroevolution(small_config(tmp_path), torch.device('cpu'))
    assert resumed._load_evolution_progress()
    assert 0.15 <= resumed.current_mutation_rate <= 0.30
    assert resumed.current_mutation_rate != 0.355
    assert resumed.population == previous.population
    resumed._save_evolution_progress()
    saved = json.loads(Path(resumed.progress_json_path).read_text())
    assert saved['runtime_state']['current_mutation_rate'] == resumed.current_mutation_rate
    assert saved['config_snapshot']['reference_injection_schedule'] == [[0, 0.0], [10, 0.10], [20, 0.20]]
