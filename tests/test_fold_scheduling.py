import json
import threading
import time
from contextlib import nullcontext

import pytest
import torch

from neuroevolution.config import get_default_config, validate_config
from neuroevolution.evolution import engine
from neuroevolution.evolution import fitness


def _fold_metrics(score):
    return {
        'accuracy': score,
        'sensitivity': score,
        'specificity': score,
        'precision': score,
        'f1_score': score,
        'auc': score,
    }


@pytest.mark.parametrize('fold_workers', [
    1,
    3,
    5,
])
def test_global_fold_queue_tracks_individuals_and_averages(
    monkeypatch, tmp_path, fold_workers
):
    config = get_default_config(info_path=str(tmp_path))
    config.update({
        'population_size': 3,
        'num_folds': 5,
        'fold_parallel_workers': fold_workers,
        'selection_strategy': 'fitness',
    })
    validate_config(config)
    evolution = engine.HybridNeuroevolution(config, torch.device('cpu'))
    evolution.population = [
        {'id': f'g{index}', 'optimizer': 'adam', 'learning_rate': 0.001}
        for index in range(1, 4)
    ]
    barrier = threading.Barrier(fold_workers, timeout=5)
    lock = threading.Lock()
    running = 0
    peak = 0
    running_by_individual = {}
    peak_per_individual = 0
    started = []

    def fake_train_fold(genome, fold_num, fold_config, device):
        nonlocal running, peak, peak_per_individual
        with lock:
            running += 1
            running_by_individual[genome['id']] = running_by_individual.get(genome['id'], 0) + 1
            peak = max(peak, running)
            peak_per_individual = max(peak_per_individual, running_by_individual[genome['id']])
            started.append((genome['id'], fold_num))
            first_wave = len(started) <= fold_workers
        if first_wave:
            barrier.wait()
        time.sleep(0.002)
        with lock:
            running -= 1
            running_by_individual[genome['id']] -= 1
        score = 10 * int(genome['id'][1:]) + fold_num
        return fold_num, float(score), object(), _fold_metrics(float(score))

    monkeypatch.setattr(engine, 'train_fold_in_thread', fake_train_fold)
    monkeypatch.setattr(engine, 'estimate_genome_parameter_count', lambda *args: 1)
    monkeypatch.setattr(engine.HybridNeuroevolution, '_format_architecture', lambda *args: 'toy')
    monkeypatch.setattr(engine.HybridNeuroevolution, 'save_best_checkpoint', lambda *args: None)

    evolution.evaluate_population()

    assert peak == fold_workers
    if fold_workers >= 3:
        assert {'g1', 'g2', 'g3'} <= {individual for individual, _ in started[:fold_workers]}
    if fold_workers == 1:
        assert peak_per_individual == 1
    assert len(started) == 15
    progress = json.loads((tmp_path / 'fold_progress.json').read_text(encoding='utf-8'))
    for index, genome in enumerate(evolution.population, start=1):
        expected_mean = 10 * index + 3
        assert genome['fitness'] == pytest.approx(expected_mean)
        entry = progress['individuals'][f'{index}:g{index}']
        assert entry['status'] == 'completed'
        assert entry['average_fitness'] == pytest.approx(expected_mean)
        assert {fold['status'] for fold in entry['folds'].values()} == {'completed'}
        assert entry['folds']['1']['fitness'] == pytest.approx(10 * index + 1)


def test_concurrency_config_rejects_zero_workers():
    config = get_default_config()
    config['fold_parallel_workers'] = 0
    with pytest.raises(ValueError, match='fold_parallel_workers'):
        validate_config(config)


def test_cuda_fold_stream_is_synchronized_before_return(monkeypatch):
    synchronized = []

    class FakeStream:
        def synchronize(self):
            synchronized.append(True)

    monkeypatch.setattr(fitness.torch.cuda, 'Stream', lambda **kwargs: FakeStream())
    monkeypatch.setattr(fitness.torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(
        fitness, 'train_fold_in_thread',
        lambda genome, fold, config, device: (fold, 50.0, object(), _fold_metrics(50.0)),
    )

    result = fitness.train_fold_on_cuda_stream({}, 1, {}, torch.device('cuda'))

    assert result[1] == 50.0
    assert synchronized == [True]


def test_global_queue_runs_real_cpu_training_and_aggregates(monkeypatch, tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self, genome, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, data):
            score = self.weight * data[:, 0]
            return torch.stack((-score, score), dim=1)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[-1.0], [1.0], [-0.5], [0.5]]),
        torch.tensor([0, 1, 0, 1]),
    )

    def fresh_loaders(*args, **kwargs):
        return (
            torch.utils.data.DataLoader(dataset, batch_size=2),
            torch.utils.data.DataLoader(dataset, batch_size=2),
        )

    config = get_default_config(info_path=str(tmp_path))
    config.update({
        'population_size': 2,
        'num_folds': 2,
        'fold_parallel_workers': 2,
        'num_epochs': 1,
        'use_amp': False,
        'selection_strategy': 'fitness',
        'fold_cache_mode': 'none',
    })
    evolution = engine.HybridNeuroevolution(config, torch.device('cpu'))
    evolution.population = [
        {'id': f'g{index}', 'optimizer': 'adam', 'learning_rate': 0.01}
        for index in range(1, 3)
    ]
    monkeypatch.setattr(fitness, 'load_fold_data', fresh_loaders)
    monkeypatch.setattr(fitness, 'EvolvableCNN', TinyModel)
    monkeypatch.setattr(engine, 'estimate_genome_parameter_count', lambda *args: 1)
    monkeypatch.setattr(engine.HybridNeuroevolution, '_format_architecture', lambda *args: 'tiny')
    monkeypatch.setattr(engine.HybridNeuroevolution, 'save_best_checkpoint', lambda *args: None)

    evolution.evaluate_population()

    assert all(genome['metrics']['n_valid_folds'] == 2 for genome in evolution.population)
    assert all(genome['fitness'] > 0 for genome in evolution.population)
    assert len(evolution.fold_progress) == 2
    assert all(entry['average_fitness'] is not None for entry in evolution.fold_progress.values())
    assert all(
        fold['epoch'] == 1 and fold['total_epochs'] == 1 and 'started_at' in fold
        for entry in evolution.fold_progress.values() for fold in entry['folds'].values()
    )


def test_fold_failure_is_visible_and_stops_evaluation(monkeypatch, tmp_path):
    config = get_default_config(info_path=str(tmp_path))
    config.update({'population_size': 2, 'num_folds': 1, 'fold_parallel_workers': 1})
    evolution = engine.HybridNeuroevolution(config, torch.device('cpu'))
    evolution.population = [
        {'id': 'g1', 'optimizer': 'adam', 'learning_rate': 0.001},
        {'id': 'g2', 'optimizer': 'adam', 'learning_rate': 0.001},
    ]

    def fail_fold(*args):
        raise torch.cuda.OutOfMemoryError('simulated OOM')

    monkeypatch.setattr(engine, 'train_fold_in_thread', fail_fold)
    with pytest.raises(torch.cuda.OutOfMemoryError, match='simulated OOM'):
        evolution.evaluate_population()

    progress = json.loads((tmp_path / 'fold_progress.json').read_text(encoding='utf-8'))
    assert progress['individuals']['1:g1']['status'] == 'failed'
    assert progress['individuals']['1:g1']['folds']['1']['status'] == 'failed'
