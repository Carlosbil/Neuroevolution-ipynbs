"""Resume from experiment artifacts without resetting previous progress."""

import copy
import json
from pathlib import Path

import pytest
import torch

from neuroevolution.config import get_default_config
from neuroevolution.evolution.engine import HybridNeuroevolution


def saved_evolution(tmp_path, *, completed=False):
    config = get_default_config(str(tmp_path))
    config.update(population_size=2, max_generations=4, fitness_threshold=101)
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    evolution.generation = 2
    evolution.last_evaluated_generation = 2 if completed else 1
    evolution.population = [{'id': 'first', 'fitness': 75}, {'id': 'second', 'fitness': 70}]
    evolution.best_individual = copy.deepcopy(evolution.population[0])
    evolution.fitness_history = [60, 75] + ([75] if completed else [])
    evolution.generation_stats = [
        {'generation': generation} for generation in range(3 if completed else 2)
    ]
    evolution.generations_without_improvement = 1
    evolution.best_fitness_overall = 75
    evolution.best_fitness_for_early_stopping = 75
    evolution.current_mutation_rate = 0.265
    evolution.current_max_conv_layers = 5
    evolution.current_max_fc_layers = 3
    evolution.template_cursor = 7
    evolution._save_evolution_progress()
    Path(evolution.generation_log_path).write_text('Previous generation log\n')
    return evolution, config


def test_default_initialization_restores_saved_population_and_runtime(tmp_path):
    previous, config = saved_evolution(tmp_path)
    before = Path(previous.progress_json_path).read_bytes()
    resumed = HybridNeuroevolution(config, torch.device('cpu'))
    assert config['resume_evolution'] is True
    resumed.initialize_population()
    for field in ('generation', 'last_evaluated_generation', 'population',
                  'best_individual', 'fitness_history', 'generation_stats',
                  'generations_without_improvement', 'best_fitness_overall',
                  'best_fitness_for_early_stopping', 'current_mutation_rate',
                  'current_max_conv_layers', 'current_max_fc_layers', 'template_cursor'):
        assert getattr(resumed, field) == getattr(previous, field)
    assert Path(resumed.progress_json_path).read_bytes() == before
    assert Path(resumed.generation_log_path).read_text() == 'Previous generation log\n'


@pytest.mark.parametrize('completed, expected_generation', [(False, 2), (True, 3)])
def test_evolve_resumes_pending_or_next_generation_without_duplicate_history(
    tmp_path, monkeypatch, completed, expected_generation
):
    _, config = saved_evolution(tmp_path, completed=completed)
    config['max_generations'] = expected_generation + 1
    resumed = HybridNeuroevolution(config, torch.device('cpu'))
    evaluated, reproduced = [], []

    def evaluate():
        evaluated.append(resumed.generation)
        resumed.generation_stats.append({'generation': resumed.generation})
        resumed.fitness_history.append(75)

    monkeypatch.setattr(resumed, 'evaluate_population', evaluate)
    monkeypatch.setattr(resumed, 'selection_and_reproduction', lambda: reproduced.append(resumed.generation))
    monkeypatch.setattr(resumed, '_update_adaptive_mutation', lambda: None)
    monkeypatch.setattr(resumed, '_print_final_metrics_summary', lambda: None)
    winner = resumed.evolve()
    assert winner['id'] == 'first'
    assert evaluated == [expected_generation]
    assert reproduced == ([3] if completed else [])
    saved = json.loads(Path(resumed.progress_json_path).read_text())
    assert [row['generation'] for row in saved['generation_stats']] == list(range(expected_generation + 1))
    assert len(saved['fitness_history']) == expected_generation + 1
    assert saved['generations_without_improvement'] == 2


@pytest.mark.parametrize('foreign_path', [
    '/home/jovyan/v3/artifacts/test_old_vowels_audio/best_model.pth',
    'C:\\previous-project\\artifacts\\test_old_vowels_audio\\best_model.pth',
])
def test_resume_relocates_dataset_and_best_checkpoint(tmp_path, foreign_path):
    previous, config = saved_evolution(tmp_path)
    progress = json.loads(Path(previous.progress_json_path).read_text())
    progress['best_checkpoint_path'] = foreign_path
    progress['config_snapshot']['data_path'] = '/home/jovyan/v3/artifacts/old_vowels_dataset'
    Path(previous.progress_json_path).write_text(json.dumps(progress))
    local_checkpoint = tmp_path / 'best_model.pth'
    local_checkpoint.write_bytes(b'previous model weights')
    resumed = HybridNeuroevolution(config, torch.device('cpu'))
    resumed.initialize_population()
    assert Path(resumed.best_checkpoint_path) == local_checkpoint
    assert local_checkpoint.read_bytes() == b'previous model weights'
    assert resumed.config == config


def test_json_only_resume_preserves_winner_and_reports_missing_weights(tmp_path, capsys):
    previous, config = saved_evolution(tmp_path)
    progress = json.loads(Path(previous.progress_json_path).read_text())
    progress['best_checkpoint_path'] = '/home/jovyan/v3/artifacts/missing.pth'
    Path(previous.progress_json_path).write_text(json.dumps(progress))
    resumed = HybridNeuroevolution(config, torch.device('cpu'))
    resumed.initialize_population()
    assert resumed.best_checkpoint_path is None
    assert resumed.best_individual == previous.best_individual
    assert 'weights are missing' in capsys.readouterr().out


@pytest.mark.parametrize('contents', ['{broken', '{}', '[]'])
def test_bad_progress_fails_without_resetting_artifacts(tmp_path, contents):
    progress_path = tmp_path / 'evolution_progress.json'
    log_path = tmp_path / 'generation_progress.txt'
    progress_path.write_text(contents)
    log_path.write_text('Previous log')
    evolution = HybridNeuroevolution(get_default_config(str(tmp_path)), torch.device('cpu'))
    with pytest.raises(ValueError, match='Existing artifacts have been preserved'):
        evolution.initialize_population()
    assert progress_path.read_text() == contents
    assert log_path.read_text() == 'Previous log'


@pytest.mark.parametrize('changes', [
    {'num_epochs': 3}, {'population_size': 3},
    {'evolution_dataset_id': 'another_dataset'}, {'data_path': '/another/dataset'},
])
def test_incompatible_resume_preserves_existing_progress(tmp_path, changes):
    previous, config = saved_evolution(tmp_path)
    before = Path(previous.progress_json_path).read_bytes()
    resumed = HybridNeuroevolution({**config, **changes}, torch.device('cpu'))
    with pytest.raises(ValueError, match='differs'):
        resumed.initialize_population()
    assert Path(previous.progress_json_path).read_bytes() == before
    assert Path(previous.generation_log_path).read_text() == 'Previous generation log\n'


def test_failed_progress_write_preserves_last_complete_json(tmp_path, monkeypatch):
    previous, _ = saved_evolution(tmp_path)
    before = Path(previous.progress_json_path).read_bytes()

    def fail_dump(data, stream, **kwargs):
        stream.write('{partial')
        raise OSError('disk full')

    monkeypatch.setattr(json, 'dump', fail_dump)
    previous.generation += 1
    with pytest.raises(OSError, match='disk full'):
        previous._save_evolution_progress()
    assert Path(previous.progress_json_path).read_bytes() == before


def test_explicit_restart_skips_even_invalid_saved_progress(tmp_path):
    config = get_default_config(str(tmp_path))
    config['resume_evolution'] = False
    Path(tmp_path / 'evolution_progress.json').write_text('{broken')
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    assert evolution._load_evolution_progress() is False


def test_absent_progress_starts_a_new_population(tmp_path):
    config = get_default_config(str(tmp_path))
    config.update(population_size=2, architecture_template_ids=[],
                  architecture_template_seed_fraction=0)
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    evolution.initialize_population()
    assert len(evolution.population) == 2
    assert evolution.generation == 0
    assert evolution.fitness_history == []
