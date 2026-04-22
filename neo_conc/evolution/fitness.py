"""
Concurrent fitness evaluation for neo_conc.

Differences vs the original `neuroevolution.evolution.fitness`:

1. Each generation evaluates the population on a SINGLE fold
   (rotated per generation by the engine). Fitness for one genome
   in one generation = single-fold F1.
2. Individuals are evaluated CONCURRENTLY in batches (default 5
   at a time) on the SAME GPU using torch.cuda.Stream() + AMP,
   reproducing the multi_cnn_single_gpu_parallel_training pattern.
3. The shared fold DataLoader is loaded once per generation and
   reused across the batch — no per-genome reload cost.

Public API:
    - evaluate_population_concurrent(population, fold_num, config, device)
    - train_individual_on_fold(genome, fold_num, train_loader, test_loader,
                               config, device, stream)
    - load_fold_data(fold_number, config, device)  # cached

Each genome receives a `metrics` dict of the SAME shape used by the
original engine so plotting/reporting code keeps working. The "_std"
fields are 0 because we only use one fold per generation.
"""

import os
import threading
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.evolvable_cnn import EvolvableCNN
from ..config import OPTIMIZERS

_FOLD_DATALOADER_CACHE: Dict[tuple, Tuple[DataLoader, DataLoader]] = {}
_FOLD_DATALOADER_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# DataLoader / fold helpers (kept compatible with the original package)
# ---------------------------------------------------------------------------

def _resolve_fold_files_directory(config: dict) -> str:
    fold_subdir = config.get(
        'fold_files_subdirectory',
        f"files_real_{config['fold_id']}"
    )
    return os.path.join(config['data_path'], fold_subdir)


def _resolve_dataloader_settings(config: dict, device: torch.device) -> Tuple[int, bool, int, bool]:
    configured_workers = config.get('dataloader_num_workers')
    if configured_workers is None:
        cpu_count = os.cpu_count() or 1
        # Workers shared across concurrent individuals (not folds anymore)
        concurrent_workers = max(
            1,
            int(config.get('concurrent_individuals', 5))
        )
        num_workers = max(1, min(4, cpu_count // concurrent_workers))
    else:
        num_workers = max(0, int(configured_workers))

    persistent_workers = bool(config.get('dataloader_persistent_workers', True)) and num_workers > 0
    prefetch_factor = max(1, int(config.get('dataloader_prefetch_factor', 2)))
    pin_memory = bool(config.get('dataloader_pin_memory', True)) and device.type == 'cuda'
    return num_workers, persistent_workers, prefetch_factor, pin_memory


def _resolve_cache_mode(config: dict) -> str:
    cache_mode = str(config.get('fold_cache_mode', 'ram')).lower()
    if cache_mode not in {'none', 'ram', 'memmap'}:
        cache_mode = 'ram'
    return cache_mode


def _build_fold_cache_key(fold_number: int, config: dict, device: torch.device, cache_mode: str):
    num_workers, persistent_workers, prefetch_factor, pin_memory = _resolve_dataloader_settings(config, device)
    return (
        os.path.abspath(_resolve_fold_files_directory(config)),
        config['dataset_id'],
        int(fold_number),
        int(config['batch_size']),
        num_workers,
        persistent_workers,
        prefetch_factor,
        pin_memory and cache_mode != 'none',
        cache_mode,
    )


def _load_numpy_array(path: str, cache_mode: str) -> np.ndarray:
    if cache_mode == 'memmap':
        return np.load(path, mmap_mode='r')
    return np.load(path)


def load_fold_data(fold_number: int, config: dict, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    """Loads (train_loader, test_loader) for a given fold, with cross-thread cache."""
    cache_mode = _resolve_cache_mode(config)
    cache_enabled = cache_mode in {'ram', 'memmap'}

    if not cache_enabled:
        return _load_fold_data_uncached(fold_number, config, device, cache_mode)

    cache_key = _build_fold_cache_key(fold_number, config, device, cache_mode)
    with _FOLD_DATALOADER_CACHE_LOCK:
        cached = _FOLD_DATALOADER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    loaded = _load_fold_data_uncached(fold_number, config, device, cache_mode)
    with _FOLD_DATALOADER_CACHE_LOCK:
        existing = _FOLD_DATALOADER_CACHE.get(cache_key)
        if existing is not None:
            return existing
        _FOLD_DATALOADER_CACHE[cache_key] = loaded
        return loaded


def _load_fold_data_uncached(
    fold_number: int,
    config: dict,
    device: torch.device,
    cache_mode: str,
) -> Tuple[DataLoader, DataLoader]:
    fold_files_directory = _resolve_fold_files_directory(config)
    dataset_id = config['dataset_id']

    x_train = _load_numpy_array(
        os.path.join(fold_files_directory, f'X_train_{dataset_id}_fold_{fold_number}.npy'),
        cache_mode,
    )
    y_train = _load_numpy_array(
        os.path.join(fold_files_directory, f'y_train_{dataset_id}_fold_{fold_number}.npy'),
        cache_mode,
    )
    x_val = _load_numpy_array(
        os.path.join(fold_files_directory, f'X_val_{dataset_id}_fold_{fold_number}.npy'),
        cache_mode,
    )
    y_val = _load_numpy_array(
        os.path.join(fold_files_directory, f'y_val_{dataset_id}_fold_{fold_number}.npy'),
        cache_mode,
    )
    x_test = _load_numpy_array(
        os.path.join(fold_files_directory, f'X_test_{dataset_id}_fold_{fold_number}.npy'),
        cache_mode,
    )
    y_test = _load_numpy_array(
        os.path.join(fold_files_directory, f'y_test_{dataset_id}_fold_{fold_number}.npy'),
        cache_mode,
    )

    if len(x_train.shape) == 2:
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    x_eval = torch.cat([x_val_tensor, x_test_tensor], dim=0)
    y_eval = torch.cat([y_val_tensor, y_test_tensor], dim=0)
    test_dataset = torch.utils.data.TensorDataset(x_eval, y_eval)

    num_workers, persistent_workers, prefetch_factor, pin_memory = _resolve_dataloader_settings(config, device)
    loader_kwargs = {
        'batch_size': config['batch_size'],
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    fold_train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    fold_test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return fold_train_loader, fold_test_loader


# ---------------------------------------------------------------------------
# Single-fold training (one individual at a time, optionally on a CUDA stream)
# ---------------------------------------------------------------------------

def _empty_metrics() -> dict:
    return {
        'accuracy': 0.0, 'sensitivity': 0.0, 'specificity': 0.0,
        'precision': 0.0, 'f1_score': 0.0, 'auc': 0.0,
    }


def train_individual_on_fold(
    genome: dict,
    fold_num: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    stream: 'torch.cuda.Stream | None' = None,
) -> Tuple[float, nn.Module, dict]:
    """
    Trains ONE genome on ONE fold. Returns (f1_score, trained_model, metrics).

    If `stream` is a torch.cuda.Stream, the forward/backward passes run inside
    that stream so multiple individuals can overlap on the same GPU
    (mirrors multi_cnn_single_gpu_parallel_training).
    """
    try:
        try:
            model = EvolvableCNN(genome, config).to(device)
        except ValueError as e:
            msg = str(e)
            if "Invalid architecture" in msg or "Expected more than 1 value per channel" in msg:
                print(f"      x Genome {genome['id']} on fold {fold_num}: invalid architecture - {msg[:120]}")
                return 0.0, None, _empty_metrics()
            raise

        optimizer_class = OPTIMIZERS[genome['optimizer']]
        optimizer = optimizer_class(model.parameters(), lr=genome['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        amp_enabled = bool(config.get('use_amp', True)) and device.type == 'cuda'
        amp_dtype_name = str(config.get('amp_dtype', 'float16')).lower()
        amp_dtype = torch.bfloat16 if amp_dtype_name == 'bfloat16' else torch.float16
        autocast_device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

        max_epochs = int(config.get('num_epochs', 30))
        patience_left = int(config.get('epoch_patience', 10))
        improvement_threshold = float(config.get('improvement_threshold', 0.01))
        validation_frequency = max(1, int(config.get('validation_frequency_epochs', 2)))

        best_acc = 0.0
        best_state = None

        for epoch_idx in range(max_epochs):
            model.train()
            batch_count = 0
            max_batches = min(
                len(train_loader),
                int(config.get('early_stopping_patience', len(train_loader)))
            )

            for data, target in train_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
                with stream_ctx:
                    with torch.autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=amp_enabled):
                        output = model(data)
                        loss = criterion(output, target)

                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                batch_count += 1
                if batch_count >= max_batches:
                    break

            should_validate = ((epoch_idx + 1) % validation_frequency == 0) or (epoch_idx + 1 == max_epochs)
            if not should_validate:
                continue

            if stream is not None:
                stream.synchronize()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    with torch.autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=amp_enabled):
                        output = model(data)
                    _, predicted = torch.max(output, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            acc = 100.0 * correct / max(1, total)
            if acc > (best_acc + improvement_threshold):
                best_acc = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_left = int(config.get('epoch_patience', 10))
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        if stream is not None:
            stream.synchronize()

        # Final metrics on the test+val pool (same as original package)
        model.eval()
        all_targets, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=amp_enabled):
                    output = model(data)
                probs = F.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                all_targets.extend(target.cpu().numpy().tolist())
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_probs.extend(probs[:, 1].cpu().numpy().tolist())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = 100.0 * (tp + tn) / max(1, len(y_true))
        sensitivity = 100.0 * tp / max(1, tp + fn)
        specificity = 100.0 * tn / max(1, tn + fp)
        precision = 100.0 * tp / max(1, tp + fp)
        f1_score = 2.0 * precision * sensitivity / max(1e-8, precision + sensitivity)

        auc = 0.0
        if len(np.unique(y_true)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_true, np.array(all_probs)) * 100.0)
            except Exception:
                auc = 0.0

        metrics = {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1_score),
            'auc': float(auc),
        }

        print(
            f"      -> Genome {genome['id']} | fold {fold_num} | "
            f"Acc={accuracy:.2f}% Sen={sensitivity:.2f}% Spe={specificity:.2f}% "
            f"F1={f1_score:.2f}% AUC={auc:.2f}%"
        )

        return float(f1_score), model, metrics

    except Exception as e:
        print(f"      ERROR training genome {genome['id']} on fold {fold_num}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, None, _empty_metrics()


# ---------------------------------------------------------------------------
# Population-level concurrent evaluation on a single fold
# ---------------------------------------------------------------------------

def _wrap_metrics_with_zero_std(metrics: dict, fold_num: int) -> dict:
    """Wraps single-fold metrics with std=0 so downstream reporting works."""
    return {
        'accuracy': metrics['accuracy'], 'accuracy_std': 0.0,
        'sensitivity': metrics['sensitivity'], 'sensitivity_std': 0.0,
        'specificity': metrics['specificity'], 'specificity_std': 0.0,
        'precision': metrics['precision'], 'precision_std': 0.0,
        'f1_score': metrics['f1_score'], 'f1_score_std': 0.0,
        'auc': metrics['auc'], 'auc_std': 0.0,
        'fold_metrics': {fold_num: metrics},
        'n_valid_folds': 1 if metrics.get('accuracy', 0) > 0 or metrics.get('f1_score', 0) > 0 else 0,
        'evaluation_fold': fold_num,
    }


def evaluate_population_concurrent(
    population: List[dict],
    fold_num: int,
    config: dict,
    device: torch.device,
) -> List[Tuple[float, nn.Module, dict]]:
    """
    Evaluates the whole population on ONE fold, training individuals
    concurrently in batches of `concurrent_individuals` (default 5).

    Returns a list of (fitness, model, aggregated_metrics) aligned with `population`.
    `aggregated_metrics` is shaped like the original 5-fold output (with std=0)
    so the engine and reporting code stay compatible.
    """
    concurrent_individuals = max(1, int(config.get('concurrent_individuals', 5)))

    print(
        f"\n   Evaluating {len(population)} individuals on FOLD {fold_num} "
        f"with concurrency={concurrent_individuals} (one CUDA stream per individual)"
    )

    # Load the fold once for the whole generation
    train_loader, test_loader = load_fold_data(fold_num, config, device)

    results: List[Tuple[float, nn.Module, dict]] = [None] * len(population)  # type: ignore

    # Process the population in batches of concurrent_individuals
    for batch_start in range(0, len(population), concurrent_individuals):
        batch = list(enumerate(population[batch_start:batch_start + concurrent_individuals]))
        batch_size = len(batch)

        # One CUDA stream per slot in this batch (only if CUDA)
        if device.type == 'cuda':
            streams = [torch.cuda.Stream() for _ in range(batch_size)]
        else:
            streams = [None] * batch_size

        print(
            f"      -> Submitting batch {batch_start // concurrent_individuals + 1} "
            f"with {batch_size} individuals (genomes: "
            f"{[g['id'] for _, g in batch]})"
        )

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_index = {}
            for slot_idx, (pop_idx, genome) in enumerate(batch):
                future = executor.submit(
                    train_individual_on_fold,
                    genome,
                    fold_num,
                    train_loader,
                    test_loader,
                    config,
                    device,
                    streams[slot_idx],
                )
                future_to_index[future] = pop_idx

            for future in as_completed(future_to_index):
                pop_idx = future_to_index[future]
                fitness, model, raw_metrics = future.result()
                aggregated = _wrap_metrics_with_zero_std(raw_metrics, fold_num)
                results[pop_idx] = (fitness, model, aggregated)

        if device.type == 'cuda':
            torch.cuda.synchronize()

    return results


# ---------------------------------------------------------------------------
# Backward-compatible single-genome wrapper (so other modules can import it)
# ---------------------------------------------------------------------------

def evaluate_fitness(
    genome: dict,
    config: dict,
    device: torch.device,
    fold_num: int = 1,
) -> Tuple[float, nn.Module, dict]:
    """
    Compatibility shim: evaluates ONE genome on ONE fold. Used when
    callers want to score an individual outside the population loop
    (e.g. the engine reusing the original interface).
    """
    train_loader, test_loader = load_fold_data(fold_num, config, device)
    fitness, model, raw_metrics = train_individual_on_fold(
        genome, fold_num, train_loader, test_loader, config, device, stream=None
    )
    return fitness, model, _wrap_metrics_with_zero_std(raw_metrics, fold_num)
