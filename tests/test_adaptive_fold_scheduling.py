import json
import threading
import time
from contextlib import nullcontext

import pytest
import torch

from neuroevolution.config import get_default_config, validate_config
from neuroevolution.evolution import engine, fitness, fold_scheduler
from neuroevolution.evolution.fold_scheduler import GIB, VramFoldScheduler, is_cuda_memory_error


def adaptive_config(**overrides):
    # Exercise automatic concurrency independently of experiment defaults.
    config = get_default_config()
    config.update(fold_auto_workers=True, fold_parallel_workers=3, fold_auto_max_workers=10)
    config.update(overrides)
    return config


def test_admission_observes_reserve_cost_growth_interval_and_backoff(monkeypatch):
    now = [0.0]
    free = [40 * GIB]
    allocated = [0]
    monkeypatch.setattr(fold_scheduler.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (free[0], 48 * GIB))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: allocated[0])
    config = adaptive_config(fold_parallel_workers=1)
    scheduler = VramFoldScheduler(config, torch.device('cuda'))
    scheduler.update(0)
    assert scheduler.admit(0)
    assert not scheduler.admit(1)
    now[0] = 30.0
    assert scheduler.admit(1)
    assert scheduler.limit == 2
    assert not scheduler.admit(2)
    allocated[0] = 16 * GIB
    free[0] = 14 * GIB
    scheduler.update(2)
    assert scheduler.estimate == 12 * GIB
    now[0] = 60.0
    assert not scheduler.admit(2)  # 12 GiB cost + 4.8 GiB reserve do not fit.
    free[0] = 40 * GIB
    scheduler.update(2)
    assert scheduler.admit(2)
    assert scheduler.limit == 3
    scheduler.backoff()
    assert not scheduler.admit(1)
    now[0] = 1000.0
    assert scheduler.admit(1)
    assert scheduler.maximum == 10
    assert scheduler.limit == 2


def test_initial_workers_are_budgeted_before_their_allocations_become_visible(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (14 * GIB, 48 * GIB))
    scheduler = VramFoldScheduler(adaptive_config(), torch.device('cuda'))
    scheduler.update(0)
    assert scheduler.limit == 3
    assert scheduler.admit(0)
    scheduler.reserve_slot()
    assert scheduler.admit(1)
    scheduler.reserve_slot()
    assert not scheduler.admit(2)
    scheduler.update(2, warming=2)
    assert not scheduler.admit(2)
    assert scheduler.snapshot()['startup_reserved_bytes'] == 8 * GIB


def test_large_architecture_estimate_expires_for_later_small_models(monkeypatch):
    now = [0.0]
    allocated = [0]
    free = [48 * GIB]
    monkeypatch.setattr(fold_scheduler.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: allocated[0])
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (free[0], 48 * GIB))
    scheduler = VramFoldScheduler(adaptive_config(), torch.device('cuda'))
    scheduler.update(0)
    allocated[0], free[0] = 24 * GIB, 24 * GIB
    scheduler.update(1)
    assert scheduler.estimate == 36 * GIB
    allocated[0], free[0] = 4 * GIB, 40 * GIB
    now[0] = 10
    scheduler.update(1)
    assert not scheduler.admit(1)
    now[0] = 61
    scheduler.update(1)
    assert scheduler.estimate == 6 * GIB
    assert scheduler.admit(1)


def test_low_memory_does_not_cancel_active_folds_or_deadlock_queue(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (GIB, 48 * GIB))
    scheduler = VramFoldScheduler(adaptive_config(), torch.device('cuda'))
    scheduler.limit = 4
    scheduler.update(4)
    assert scheduler.limit == 3
    assert not scheduler.admit(4)
    assert not scheduler.admit(1)
    assert scheduler.admit(0)


def test_telemetry_failure_falls_back_to_one_fold(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    def unavailable(device):
        raise RuntimeError('telemetry unavailable')
    monkeypatch.setattr(torch.cuda, 'mem_get_info', unavailable)
    scheduler = VramFoldScheduler(adaptive_config(), torch.device('cuda'))
    scheduler.limit = 3
    scheduler.update(2)
    assert scheduler.limit == 1
    assert scheduler.admit(0)
    assert not scheduler.admit(1)
    assert scheduler.snapshot()['free_vram_bytes'] is None


@pytest.mark.parametrize('message,expected', [
    ('CUDA out of memory', True),
    ('NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1165', True),
    ('CUDA device-side assert triggered', False),
    ('Invalid architecture', False),
    ('NVML_SUCCESS == r in some other component', False),
])
def test_memory_error_classification(message, expected):
    assert is_cuda_memory_error(RuntimeError(message)) is expected


def _evolution(tmp_path, **overrides):
    config = get_default_config(info_path=str(tmp_path))
    config.update({'num_folds': 2, 'fold_auto_growth_interval_s': 0.001,
                   'fold_auto_workers': True, 'fold_parallel_workers': 3,
                   'fold_auto_max_workers': 3, **overrides})
    evolution = engine.HybridNeuroevolution(config, torch.device('cuda'))
    evolution.population = [{'id': f'g{i}', 'optimizer': 'adam', 'learning_rate': 0.001}
                            for i in range(4)]
    return evolution


def _mock_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    # Observe memory while a fold is still running, without a five second delay.
    original_wait = engine.wait
    monkeypatch.setattr(engine, 'wait', lambda futures, **kw:
                        original_wait(futures, timeout=0.01, return_when=kw['return_when']))


@pytest.mark.parametrize('free_gib,expected_peak', [(40, 3), (14, 2), (6, 1)])
def test_cuda_queue_grows_only_when_vram_fits(monkeypatch, tmp_path, free_gib, expected_peak):
    _mock_cuda(monkeypatch)
    lock = threading.Lock()
    running = peak = 0
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (free_gib * GIB, 48 * GIB))
    def train(genome, fold, config, device):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.08)
        with lock:
            running -= 1
        metrics = {key: 75.0 for key in ('accuracy', 'sensitivity', 'specificity',
                                        'precision', 'f1_score', 'auc')}
        return fold, 75.0, torch.nn.Linear(1, 1), metrics
    monkeypatch.setattr(engine, 'train_fold_on_cuda_stream', train)
    evolution = _evolution(tmp_path)
    results = list(evolution._iter_population_results())
    assert len(results) == 4
    assert peak == expected_peak
    assert all(result[1] == 75.0 for result in results)
    progress = json.loads((tmp_path / 'fold_progress.json').read_text())
    assert progress['scheduler']['total_vram_bytes'] == 48 * GIB
    assert all(fold['status'] == 'completed' for entry in progress['individuals'].values()
               for fold in entry['folds'].values())


def test_initial_wave_is_concurrent_even_for_folds_shorter_than_growth_interval(monkeypatch, tmp_path):
    _mock_cuda(monkeypatch)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (40 * GIB, 48 * GIB))
    barrier = threading.Barrier(3, timeout=2)
    lock = threading.Lock()
    started = []
    cleanups = []
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: cleanups.append(True))

    def train(genome, fold, config, device):
        with lock:
            started.append((genome['id'], fold))
            initial_wave = len(started) <= 3
        if initial_wave:
            barrier.wait()
        metrics = {key: 75.0 for key in ('accuracy', 'sensitivity', 'specificity',
                                        'precision', 'f1_score', 'auc')}
        return fold, 75.0, torch.nn.Linear(1, 1), metrics

    monkeypatch.setattr(engine, 'train_fold_on_cuda_stream', train)
    evolution = _evolution(tmp_path, fold_auto_growth_interval_s=30.0)
    assert len(list(evolution._iter_population_results())) == 4
    assert len(started) == 8
    assert cleanups == []


@pytest.mark.parametrize('error_factory', [
    lambda: torch.cuda.OutOfMemoryError('simulated OOM'),
    lambda: RuntimeError('NVML_SUCCESS == r at CUDACachingAllocator.cpp:1165'),
])
def test_cuda_queue_drains_and_retries_memory_failures(monkeypatch, tmp_path, error_factory):
    _mock_cuda(monkeypatch)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (40 * GIB, 48 * GIB))
    lock = threading.Lock()
    attempts = {}
    running = 0
    retry_running_counts = []
    cleanup_running_counts = []
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: cleanup_running_counts.append(running))
    def train(genome, fold, config, device):
        nonlocal running
        key = genome['id'], fold
        with lock:
            attempts[key] = attempts.get(key, 0) + 1
            running += 1
            if attempts[key] > 1:
                retry_running_counts.append(running)
        try:
            time.sleep(0.04)
            if key == ('g1', 1) and attempts[key] == 1:
                raise error_factory()
            metrics = {key: 75.0 for key in ('accuracy', 'sensitivity', 'specificity',
                                            'precision', 'f1_score', 'auc')}
            return fold, 75.0, torch.nn.Linear(1, 1), metrics
        finally:
            with lock:
                running -= 1
    monkeypatch.setattr(engine, 'train_fold_on_cuda_stream', train)
    evolution = _evolution(tmp_path)
    results = list(evolution._iter_population_results())
    assert len(results) == 4
    assert attempts['g1', 1] == 2
    assert retry_running_counts == [1]
    assert cleanup_running_counts
    assert all(count == 0 for count in cleanup_running_counts)
    assert all(result[1] == 75.0 for result in results)
    progress = json.loads((tmp_path / 'fold_progress.json').read_text())
    assert progress['individuals']['2:g1']['folds']['1']['retries'] == 1
    assert progress['scheduler']['maximum_workers'] == 3


def test_memory_drain_refreshes_telemetry_while_a_healthy_fold_trains(monkeypatch, tmp_path):
    _mock_cuda(monkeypatch)
    free_gib = [40]
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (free_gib[0] * GIB, 48 * GIB))
    original_backoff = VramFoldScheduler.backoff

    def backoff(scheduler):
        original_backoff(scheduler)
        free_gib[0] = 8

    monkeypatch.setattr(VramFoldScheduler, 'backoff', backoff)
    healthy_started = threading.Event()
    telemetry_refreshed = threading.Event()
    snapshots = []
    attempts = {}
    evolution = _evolution(tmp_path, num_folds=1, fold_parallel_workers=2,
                           fold_auto_growth_interval_s=30.0)
    evolution.population = evolution.population[:2]
    original_save = evolution._save_fold_progress

    def save():
        snapshot = evolution.fold_scheduler_progress
        if snapshot and snapshot.get('draining'):
            snapshots.append(dict(snapshot))
            if snapshot['free_vram_bytes'] == 8 * GIB:
                free_gib[0] = 12
            elif snapshot['free_vram_bytes'] == 12 * GIB:
                telemetry_refreshed.set()
        original_save()

    evolution._save_fold_progress = save

    def train(genome, fold, config, device):
        attempts[genome['id']] = attempts.get(genome['id'], 0) + 1
        if genome['id'] == 'g0' and attempts['g0'] == 1:
            assert healthy_started.wait(timeout=3)
            raise torch.cuda.OutOfMemoryError('simulated OOM')
        if genome['id'] == 'g1':
            config['_fold_progress_callback'](1, 100)
            healthy_started.set()
            assert telemetry_refreshed.wait(timeout=3), 'telemetry froze during drain'
        metrics = {key: 75.0 for key in ('accuracy', 'sensitivity', 'specificity',
                                        'precision', 'f1_score', 'auc')}
        return fold, 75.0, torch.nn.Linear(1, 1), metrics

    monkeypatch.setattr(engine, 'train_fold_on_cuda_stream', train)
    results = list(evolution._iter_population_results())
    assert len(results) == 2 and all(result[1] == 75.0 for result in results)
    assert attempts == {'g0': 2, 'g1': 1}
    assert {snapshot['free_vram_bytes'] for snapshot in snapshots} == {8 * GIB, 12 * GIB}
    assert all(snapshot['block_reason'] == 'draining_after_memory_error' for snapshot in snapshots)
    assert snapshots[-1]['oom_cooldown_seconds'] < snapshots[0]['oom_cooldown_seconds']
    assert snapshots[-1]['startup_reserved_bytes'] == 0


@pytest.mark.parametrize('auto_workers,max_retries,attempts_per_fold', [
    (True, 1, 2), (True, 0, 1), (False, 1, 1),
])
def test_exhausted_memory_retries_assign_zero_and_continue(
    monkeypatch, tmp_path, auto_workers, max_retries, attempts_per_fold
):
    _mock_cuda(monkeypatch)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (40 * GIB, 48 * GIB))
    attempts = []
    def fail(*args):
        attempts.append(True)
        raise torch.cuda.OutOfMemoryError('too large even alone')
    monkeypatch.setattr(engine, 'train_fold_on_cuda_stream', fail)
    evolution = _evolution(tmp_path, fold_auto_workers=auto_workers,
                           fold_oom_max_retries=max_retries, fold_parallel_workers=1,
                           fold_auto_growth_interval_s=30.0)
    results = list(evolution._iter_population_results())
    assert len(results) == len(evolution.population)
    assert all(result[1] == 0.0 and result[2] is None for result in results)
    assert all(result[3]['n_valid_folds'] == 0 for result in results)
    assert len(attempts) == 8 * attempts_per_fold
    progress = json.loads((tmp_path / 'fold_progress.json').read_text())
    failed = progress['individuals']['1:g0']['folds']['1']
    assert failed['status'] == 'failed'
    assert failed['fitness'] == 0.0
    assert failed['memory_error'] == 'too large even alone'
    assert all(entry['status'] == 'completed_with_failed_folds'
               and entry['average_fitness'] == 0.0
               for entry in progress['individuals'].values())


@pytest.mark.parametrize('require_all_folds,expected_fitness', [(True, 0.0), (False, 37.5)])
def test_memory_failure_does_not_skip_other_folds_or_individuals(
    monkeypatch, tmp_path, require_all_folds, expected_fitness
):
    _mock_cuda(monkeypatch)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (40 * GIB, 48 * GIB))
    calls = []

    def train(genome, fold, config, device):
        calls.append((genome['id'], fold))
        if (genome['id'], fold) == ('g0', 1):
            raise torch.cuda.OutOfMemoryError('too large')
        metrics = {key: 75.0 for key in ('accuracy', 'sensitivity', 'specificity',
                                        'precision', 'f1_score', 'auc')}
        return fold, 75.0, torch.nn.Linear(1, 1), metrics

    monkeypatch.setattr(engine, 'train_fold_on_cuda_stream', train)
    evolution = _evolution(tmp_path, fold_oom_max_retries=0, require_all_folds=require_all_folds)
    results = {result[0]: result for result in evolution._iter_population_results()}
    assert len(calls) == 8
    assert results[0][1] == expected_fitness
    assert results[0][3]['n_valid_folds'] == 1
    assert all(results[index][1] == 75.0 for index in (1, 2, 3))


def test_nvml_error_escapes_training_instead_of_becoming_zero(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('NVML_SUCCESS == r at CUDACachingAllocator.cpp:1165')
    monkeypatch.setattr(fitness, 'load_fold_data', fail)
    with pytest.raises(RuntimeError, match='NVML_SUCCESS'):
        fitness.train_fold_in_thread({}, 1, get_default_config(), torch.device('cpu'))


@pytest.mark.parametrize('key,value', [
    ('fold_auto_workers', 'yes'), ('fold_auto_max_workers', 0),
    ('fold_auto_growth_interval_s', 0), ('fold_vram_reserve_gib', -1),
    ('fold_vram_reserve_fraction', 1), ('fold_vram_initial_fold_gib', 0),
    ('fold_vram_safety_factor', 0.5), ('fold_oom_max_retries', -1),
])
def test_invalid_adaptive_config_is_rejected(key, value):
    config = get_default_config()
    config[key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)
