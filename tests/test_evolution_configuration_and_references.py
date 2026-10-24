"""Regressions for immutable config, stopping, inherited costs and full networks."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from neuroevolution.config import get_default_config, get_operator_config, REFERENCE_ARCHITECTURE_IDS, configured_epochs
from neuroevolution.evolution.engine import HybridNeuroevolution
from neuroevolution.evolution.fitness import aggregate_fold_results
from neuroevolution.genetics.architecture_templates import create_template_genome
from neuroevolution.genetics.crossover import crossover_genomes
from neuroevolution.genetics.genome import create_random_genome
from neuroevolution.genetics.mutation import mutate_genome
from neuroevolution.models.evolvable_cnn import EvolvableCNN
from neuroevolution.models.genome_validator import estimate_genome_parameter_count
from neuroevolution.models.reference_architectures import classification_loss


def test_convergence_polling_counts_only_completed_generations(tmp_path):
    config = get_default_config(str(tmp_path))
    config['early_stopping_generations'] = 3
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    evolution.best_individual = {'fitness': 75.0}
    evolution._update_convergence_state()
    for generation in range(1, 4):
        evolution.generation = generation
        for _ in range(10):
            evolution.check_convergence()
        evolution._update_convergence_state()
        evolution._update_convergence_state()
        assert evolution.generations_without_improvement == generation
    assert evolution.check_convergence()


def test_engine_state_and_reproduction_never_overwrite_configuration(tmp_path):
    config = get_default_config(str(tmp_path))
    config.update(population_size=6, architecture_template_seed_fraction=0,
                  architecture_template_ids=[], reference_injection_fraction=0,
                  max_filters=16, min_fc_nodes=8, max_fc_nodes=32, max_conv_layers=4,
                  max_fc_layers=2, initial_max_fc_layers=1)
    before = copy.deepcopy(config)
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    evolution.initialize_population()
    for index, genome in enumerate(evolution.population):
        genome['fitness'] = 70 + index
        genome['metrics'] = {'f1_score': 70 + index}
    evolution.generation_stats = [{'std_fitness': 2.0}]
    evolution._update_adaptive_mutation()
    evolution.generation = 20
    evolution._update_incremental_complexity()
    evolution.selection_and_reproduction()
    assert config == before
    assert evolution.config == before
    assert len(evolution.population) == 6
    assert evolution.species


@pytest.mark.parametrize('cross', [0.0, 1.0])
def test_offspring_invalidate_all_evaluation_and_selection_metadata(cross):
    config = get_default_config()
    config.update(architecture_template_ids=[], architecture_template_mutation_weight=0,
                  min_filters=8, max_filters=16, min_fc_nodes=8, max_fc_nodes=32,
                  max_conv_layers=4, max_fc_layers=2, crossover_rate=cross)
    parent = create_random_genome(config)
    stale = {'parameter_count': 123, 'evaluation_time_seconds': 456,
             'pareto_rank': 0, 'crowding_distance': 99, 'metrics': {'f1_score': 80},
             'selection_objectives': {'fitness': 80}, 'skip_next_evaluation': True}
    parent.update(stale)
    child, _ = crossover_genomes(parent, parent, config)
    mutated = mutate_genome(parent, config)
    for offspring in (child, mutated):
        assert not set(stale).intersection(offspring)
        assert offspring['fitness'] == 0
    assert parent['parameter_count'] == 123


def test_new_evaluation_recalculates_even_if_stale_parent_cost_is_present(tmp_path):
    config = get_default_config(str(tmp_path))
    config.update(max_filters=16, min_fc_nodes=8, max_fc_nodes=32, max_conv_layers=4, max_fc_layers=2)
    genome = create_random_genome(config)
    genome['parameter_count'] = 1
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    evolution._record_cost_metadata(genome, {}, elapsed_seconds=1)
    model = EvolvableCNN(genome, config)
    assert genome['parameter_count'] == sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize('architecture_id', REFERENCE_ARCHITECTURE_IDS)
@pytest.mark.parametrize('modality', ['audio', 'spectrogram'])
def test_complete_library_is_not_clipped_to_genetic_bounds(architecture_id, modality):
    config = get_default_config()
    config['input_modality'] = modality
    config.update(max_conv_layers=1, max_filters=8, max_fc_nodes=8, max_model_parameters=100)
    genome = create_template_genome(architecture_id, config)
    with torch.device('meta'):
        model = EvolvableCNN(genome, config)
    assert estimate_genome_parameter_count(genome, config) == sum(parameter.numel() for parameter in model.parameters())
    assert estimate_genome_parameter_count(genome, config) > config['max_model_parameters']
    assert model.architecture_id == architecture_id


def test_vgg16_and_resnet50_keep_original_hidden_layers_and_blocks():
    config = get_default_config()
    with torch.device('meta'):
        vgg = EvolvableCNN(create_template_genome('vgg16', config), config).model
        resnet = EvolvableCNN(create_template_genome('resnet50', config), config).model
    assert [layer.out_features for layer in vgg.classifier if isinstance(layer, nn.Linear)] == [4096, 4096, 2]
    assert [len(getattr(resnet, f'layer{i}')) for i in range(1, 5)] == [3, 4, 6, 3]
    assert resnet.layer4[-1].conv3.out_channels == 2048


@pytest.mark.parametrize('architecture_id', ['lenet5', 'alexnet', 'resnet18', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', 'squeezenet1_0'])
def test_reference_families_forward_on_spectrograms(architecture_id):
    torch.set_num_threads(1)
    config = get_default_config()
    config['input_modality'] = 'spectrogram'
    model = EvolvableCNN(create_template_genome(architecture_id, config), config).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 64, 648)).shape == (1, 2)


@pytest.mark.parametrize('architecture_id', ['googlenet', 'inception_v3'])
@pytest.mark.parametrize('modality', ['audio', 'spectrogram'])
def test_auxiliary_reference_branches_are_retained_and_trained(architecture_id, modality):
    torch.set_num_threads(1)
    config = get_default_config()
    config['input_modality'] = modality
    model = EvolvableCNN(create_template_genome(architecture_id, config), config).train()
    logits = model(torch.randn(2, 1, 512) if modality == 'audio' else torch.randn(2, 64, 648))
    assert model.auxiliary_logits
    loss = classification_loss(model, logits, torch.tensor([0, 1]), nn.CrossEntropyLoss(), config)
    loss.backward()
    auxiliary = model.model.AuxLogits if architecture_id == 'inception_v3' else model.model.aux1
    assert any(parameter.grad is not None for parameter in auxiliary.parameters())
    assert logits.shape == (2, 2)


def test_reference_genetic_operators_preserve_full_architecture_and_clear_costs():
    config = get_default_config()
    config.update(architecture_template_mutation_weight=0, current_mutation_rate=1)
    parent = create_template_genome('resnet50', config)
    parent.update(parameter_count=1, metrics={'f1_score': 80}, skip_next_evaluation=True)
    child, _ = crossover_genomes(parent, parent, config)
    mutated = mutate_genome(parent, config)
    assert child['reference_architecture_id'] == mutated['reference_architecture_id'] == 'resnet50'
    assert 'parameter_count' not in child and 'parameter_count' not in mutated
    assert 'metrics' not in child and 'metrics' not in mutated


def test_collapsed_positive_classifier_cannot_get_rewarded_for_cheap_cost():
    config = get_default_config()
    metrics = dict(accuracy=50, sensitivity=100, specificity=0, precision=50,
                   f1_score=100 * 2 / 3, auc=50)
    folds = {fold: (metrics['f1_score'], object(), metrics.copy()) for fold in range(1, 6)}
    score, _, aggregated = aggregate_fold_results({'id': 'collapsed'}, config, folds)
    assert score == 0
    assert aggregated['raw_fitness'] == pytest.approx(100 * 2 / 3)
    assert aggregated['fitness_feasible'] is False


def test_complete_seeds_follow_config_optimizer_and_do_not_share_species(tmp_path):
    config = get_default_config(str(tmp_path))
    config.update(reference_seed_optimizer='sgd', reference_seed_learning_rate=0.03)
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    first = create_template_genome('resnet18', config)
    second = create_template_genome('vgg16', config)
    assert first['optimizer'] == 'sgd' and first['learning_rate'] == 0.03
    evolution.population = [first, second]
    evolution._speciate_population()
    assert len(evolution.species) == 2


def test_stagnation_boost_does_not_reduce_base_mutation_for_unique_genomes(tmp_path):
    config = get_default_config(str(tmp_path))
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    evolution.population = [create_template_genome(name, config) for name in ('lenet5', 'resnet18')]
    evolution.generation_stats = [{'std_fitness': 0}]
    evolution._update_adaptive_mutation()
    assert evolution.current_mutation_rate == config['base_mutation_rate']
    evolution.generations_without_improvement = config['early_stopping_generations']
    evolution._update_adaptive_mutation()
    assert evolution.current_mutation_rate > config['base_mutation_rate']


def test_epoch_arguments_cannot_override_config():
    assert configured_epochs({'num_epochs': 7}) == 7
    assert configured_epochs({'num_epochs': 7}, 7) == 7
    with pytest.raises(ValueError, match='cannot override'):
        configured_epochs({'num_epochs': 7}, 100)


def test_failed_checkpoint_save_preserves_previous_weights(monkeypatch, tmp_path):
    config = get_default_config(str(tmp_path))
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    model = nn.Linear(2, 2)
    evolution.save_best_checkpoint({'id': 'first', 'fitness': 70}, model)
    previous = Path(evolution.best_checkpoint_path)
    before = previous.read_bytes()
    def fail_save(*args, **kwargs):
        raise OSError('disk failure')
    monkeypatch.setattr(torch, 'save', fail_save)
    with pytest.raises(RuntimeError, match='Could not save checkpoint'):
        evolution.save_best_checkpoint({'id': 'second', 'fitness': 80}, model)
    assert Path(evolution.best_checkpoint_path) == previous
    assert previous.read_bytes() == before


@pytest.mark.parametrize('modality', ['audio', 'spectrogram'])
def test_real_evolution_trains_complete_network_and_resumes_without_config_mutation(tmp_path, modality):
    """Exercise real CPU training, aggregation, reproduction, saving and resume."""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    folder = tmp_path / 'folds'
    folder.mkdir()
    targets = np.array([0, 1] * 4, dtype=np.int64)
    shape = (8, 1, 64) if modality == 'audio' else (8, 16, 32)
    features = np.random.default_rng(42).normal(size=shape).astype(np.float32)
    features += (targets * 2 - 1).astype(np.float32)[:, None, None]
    for split in ('train', 'val'):
        np.save(folder / f'X_{split}_fixture_fold_1.npy', features)
        np.save(folder / f'y_{split}_fixture_fold_1.npy', targets)
    config = get_default_config(str(tmp_path / 'run'))
    config['input_modality'] = modality
    config.update(population_size=6, max_generations=2, fitness_threshold=101,
                  data_path=str(tmp_path), evolution_dataset_id='fixture',
                  evolution_fold_files_subdirectory='folds', num_folds=1,
                  num_frequency_bins=16, sequence_length=shape[-1], pre_fc_pool_shape=(2, 2),
                  min_filters=4, max_filters=4, min_fc_nodes=4, max_fc_nodes=4,
                  max_conv_layers=2, initial_max_conv_layers=1,
                  max_fc_layers=1, num_epochs=2, epoch_patience=2,
                  batch_size=4, fold_parallel_workers=1, fold_cache_mode='none',
                  architecture_template_ids=['lenet5'],
                  reference_injection_schedule=None,
                  random_immigrant_fraction=0.2, reference_injection_fraction=0.2,
                  use_amp=False, dataloader_pin_memory=False)
    before = copy.deepcopy(config)
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    evolution.evolve()
    assert config == before == evolution.config
    assert [row['generation'] for row in evolution.generation_stats] == [0, 1]
    assert evolution.generation == evolution.last_evaluated_generation == 1
    assert all(row['metrics']['n_valid_folds'] == 1
               for stat in evolution.generation_stats for row in stat['individual_metrics'])
    reference = create_template_genome('lenet5', config)
    model = EvolvableCNN(reference, config).eval()
    reference['fitness'] = 50
    evolution.save_best_checkpoint(reference, model)
    _, restored = evolution.load_best_checkpoint()
    data = torch.from_numpy(features)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(data), restored(data))
    saved = json.loads(Path(evolution.progress_json_path).read_text())
    assert saved['last_evaluated_generation'] == 1
    resumed = HybridNeuroevolution({**config, 'resume_evolution': True}, torch.device('cpu'))
    assert resumed._load_evolution_progress()
    assert resumed.check_convergence()
    changed = HybridNeuroevolution({**config, 'resume_evolution': True, 'num_epochs': 3}, torch.device('cpu'))
    with pytest.raises(ValueError, match='differs'):
        changed._load_evolution_progress()
