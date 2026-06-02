"""
Fitness evaluation module for parallel 5-fold cross-validation.

This module implements parallel training and evaluation of genomes using ThreadPoolExecutor
to run all 5 folds simultaneously. Fitness is calculated as the average F1-score across folds.
"""

import os
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

from ..models.evolvable_cnn import EvolvableCNN
from ..config import OPTIMIZERS, SUPPORTED_METRIC_NAMES, canonical_metric_name

_FOLD_DATALOADER_CACHE = {}
_FOLD_DATALOADER_CACHE_LOCK = threading.Lock()
_VALID_EVAL_SPLITS = {'validation', 'test', 'validation_and_test', 'all'}


def resolve_metric_name(config: dict, key: str = 'fold_selection_metric') -> str:
    """Resolves a configured metric name, including aliases and fitness indirection."""
    default_metric = 'fitness_metric' if key == 'fold_selection_metric' else 'f1_score'
    configured_metric = str(config.get(key, default_metric)).strip().lower()

    if key == 'fold_selection_metric' and configured_metric == 'fitness_metric':
        configured_metric = str(config.get('fitness_metric', 'f1_score')).strip().lower()

    if configured_metric not in SUPPORTED_METRIC_NAMES:
        valid_options = ', '.join(sorted(SUPPORTED_METRIC_NAMES | {'fitness_metric'}))
        raise ValueError(f"{key} must be one of: {valid_options}")

    return canonical_metric_name(configured_metric)


def resolve_metric_improvement_threshold(config: dict) -> float:
    """Returns the selected-metric improvement threshold with legacy fallback."""
    threshold = config.get('metric_improvement_threshold')
    if threshold is None:
        threshold = config.get('improvement_threshold', 0.01)
    return float(threshold)


def metric_value(metrics: dict, metric_name: str) -> float:
    """Fetches a numeric metric value, returning 0.0 for missing or invalid values."""
    raw_name = str(metric_name).strip().lower()
    canonical_name = canonical_metric_name(raw_name)
    value = metrics.get(canonical_name)
    if value is None and raw_name in metrics:
        value = metrics.get(raw_name)

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(numeric_value):
        return 0.0
    return numeric_value


def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Computes binary classification metrics on the same percentage scale."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0:
        return {
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'precision': 0.0,
            'f1_score': 0.0,
            'auc': 0.0,
        }

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
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            y_prob = np.asarray(y_prob)
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, y_prob) * 100.0)
        except Exception:
            auc = 0.0

    return {
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1_score),
        'auc': float(auc),
    }


def _normalize_eval_split(eval_split: str) -> str:
    """Returns a validated fold evaluation split mode."""
    normalized = str(eval_split).lower()
    if normalized not in _VALID_EVAL_SPLITS:
        valid_options = ', '.join(sorted(_VALID_EVAL_SPLITS))
        raise ValueError(f"Invalid eval_split '{eval_split}'. Expected one of: {valid_options}")
    return normalized


def _resolve_fold_files_directory(config: dict) -> str:
    """Resolves fold directory from config, preferring explicit subdirectory when available."""
    fold_subdir = config.get('fold_files_subdirectory', f"files_real_{config['fold_id']}")
    return os.path.join(config['data_path'], fold_subdir)


def _resolve_dataloader_settings(config: dict, device: torch.device) -> Tuple[int, bool, int, bool]:
    """
    Resolves DataLoader worker/prefetch settings.

    Returns:
        Tuple of (num_workers, persistent_workers, prefetch_factor, pin_memory)
    """
    configured_workers = config.get('dataloader_num_workers')
    if configured_workers is None:
        cpu_count = os.cpu_count() or 1
        fold_workers = max(1, int(config.get('fold_parallel_workers', config.get('num_folds', 5))))
        num_workers = max(1, min(4, cpu_count // fold_workers))
    else:
        num_workers = max(0, int(configured_workers))

    persistent_workers = bool(config.get('dataloader_persistent_workers', True)) and num_workers > 0
    prefetch_factor = max(1, int(config.get('dataloader_prefetch_factor', 2)))
    pin_memory = bool(config.get('dataloader_pin_memory', True)) and device.type == 'cuda'
    return num_workers, persistent_workers, prefetch_factor, pin_memory


def _resolve_cache_mode(config: dict) -> str:
    """Returns validated fold cache mode."""
    cache_mode = str(config.get('fold_cache_mode', 'ram')).lower()
    if cache_mode not in {'none', 'ram', 'memmap'}:
        cache_mode = 'ram'
    return cache_mode


def _build_fold_cache_key(
    fold_number: int,
    config: dict,
    device: torch.device,
    cache_mode: str,
    eval_split: str,
) -> Tuple[str, str, int, int, int, bool, int, bool, str, str]:
    """Builds a deterministic cache key for a fold DataLoader pair."""
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
        _normalize_eval_split(eval_split),
    )


def _load_numpy_array(path: str, cache_mode: str) -> np.ndarray:
    """Loads a numpy array with optional memmap mode."""
    if cache_mode == 'memmap':
        return np.load(path, mmap_mode='r')
    return np.load(path)


def evaluate_fitness(genome: dict, config: dict, device: torch.device) -> Tuple[float, nn.Module, dict]:
    """
    Evalua el fitness de un genoma usando 5-fold cross-validation PARALELO.
    Los 5 folds se entrenan en threads separados y se espera a que terminen todos.
    El fitness final es el promedio de la métrica de fitness configurada.

    Args:
        genome: Genome dictionary defining the architecture
        config: Configuration dictionary
        device: PyTorch device (CPU or CUDA)

    Returns:
        Tuple de (fitness, model, metrics) donde:
            - fitness: promedio de la métrica configurada de los 5 folds
            - model: modelo entrenado en el mejor fold (para checkpoint)
            - metrics: diccionario con metricas agregadas de todos los folds
    """
    print(f"      Training/Evaluating model {genome['id']} with PARALLEL 5-FOLD CROSS-VALIDATION")

    fold_scores = {}
    fold_models = {}
    fold_metrics = {}
    fitness_metric = 'f1_score'
    selection_metric = 'f1_score'

    try:
        fitness_metric = resolve_metric_name(config, 'fitness_metric')
        selection_metric = resolve_metric_name(config, 'fold_selection_metric')
        num_folds = int(config.get('num_folds', 5))
        fold_workers = max(1, min(int(config.get('fold_parallel_workers', num_folds)), num_folds))

        # Usar ThreadPoolExecutor para ejecutar folds en paralelo
        with ThreadPoolExecutor(max_workers=fold_workers) as executor:
            # Enviar folds a threads separados
            print(f"      -> Submitting {num_folds} folds to thread pool (workers={fold_workers})...")
            futures = {
                executor.submit(train_fold_in_thread, genome, fold_num, config, device): fold_num
                for fold_num in range(1, num_folds + 1)
            }

            # Esperar a que todos los folds terminen
            print(f"      -> Waiting for all {num_folds} folds to complete...")
            for future in as_completed(futures):
                fold_num, fold_score, model, metrics = future.result()
                fold_scores[fold_num] = fold_score
                fold_models[fold_num] = model
                fold_metrics[fold_num] = metrics

        # Ordenar resultados por fold_num
        sorted_folds = sorted(fold_scores.keys())
        fold_scores_list = [fold_scores[f] for f in sorted_folds]

        # Encontrar el mejor modelo
        best_fold_num = max(fold_scores, key=fold_scores.get)
        best_fold_score = fold_scores[best_fold_num]
        best_model = fold_models[best_fold_num]

        # Calcular fitness como promedio de los 5 folds
        avg_fitness = np.mean(fold_scores_list)
        std_fitness = np.std(fold_scores_list)

        # Agregar metricas de todos los folds (solo los folds validos)
        valid_metrics = [m for m in fold_metrics.values() if m is not None]

        if valid_metrics:
            aggregated_metrics = {
                'accuracy': np.mean([m['accuracy'] for m in valid_metrics]),
                'accuracy_std': np.std([m['accuracy'] for m in valid_metrics]),
                'sensitivity': np.mean([m['sensitivity'] for m in valid_metrics]),
                'sensitivity_std': np.std([m['sensitivity'] for m in valid_metrics]),
                'specificity': np.mean([m['specificity'] for m in valid_metrics]),
                'specificity_std': np.std([m['specificity'] for m in valid_metrics]),
                'precision': np.mean([m['precision'] for m in valid_metrics]),
                'precision_std': np.std([m['precision'] for m in valid_metrics]),
                'f1_score': np.mean([m['f1_score'] for m in valid_metrics]),
                'f1_score_std': np.std([m['f1_score'] for m in valid_metrics]),
                'auc': np.mean([m['auc'] for m in valid_metrics]),
                'auc_std': np.std([m['auc'] for m in valid_metrics]),
                'fold_metrics': fold_metrics,
                'n_valid_folds': len(valid_metrics),
                'fitness_split': 'validation',
                'fitness_metric': fitness_metric,
                'selection_metric': selection_metric,
            }
        else:
            aggregated_metrics = {
                'accuracy': 0.0, 'accuracy_std': 0.0,
                'sensitivity': 0.0, 'sensitivity_std': 0.0,
                'specificity': 0.0, 'specificity_std': 0.0,
                'precision': 0.0, 'precision_std': 0.0,
                'f1_score': 0.0, 'f1_score_std': 0.0,
                'auc': 0.0, 'auc_std': 0.0,
                'fold_metrics': {},
                'n_valid_folds': 0,
                'fitness_split': 'validation',
                'fitness_metric': fitness_metric,
                'selection_metric': selection_metric,
            }

        print(f"      + PARALLEL 5-Fold CV Results for {genome['id']}:")
        print(f"        Fold validation {fitness_metric} scores: {[f'{score:.2f}%' for score in fold_scores_list]}")
        print(f"        Average validation fitness: {avg_fitness:.2f}% +/- {std_fitness:.2f}%")
        print(f"        Best fold: Fold {best_fold_num} with {best_fold_score:.2f}% validation {fitness_metric}")
        print("        --- AGGREGATED METRICS ---")
        print(f"        Accuracy:     {aggregated_metrics['accuracy']:.2f}% +/- {aggregated_metrics['accuracy_std']:.2f}%")
        print(f"        Sensitivity:  {aggregated_metrics['sensitivity']:.2f}% +/- {aggregated_metrics['sensitivity_std']:.2f}%")
        print(f"        Specificity:  {aggregated_metrics['specificity']:.2f}% +/- {aggregated_metrics['specificity_std']:.2f}%")
        print(f"        Precision:    {aggregated_metrics['precision']:.2f}% +/- {aggregated_metrics['precision_std']:.2f}%")
        print(f"        F1-Score:     {aggregated_metrics['f1_score']:.2f}% +/- {aggregated_metrics['f1_score_std']:.2f}%")
        print(f"        AUC:          {aggregated_metrics['auc']:.2f}% +/- {aggregated_metrics['auc_std']:.2f}%")

        return avg_fitness, best_model, aggregated_metrics

    except Exception as e:
        print(f"      ERROR evaluating genome {genome['id']}: {e}")
        import traceback
        traceback.print_exc()
        empty_metrics = {
            'accuracy': 0.0, 'accuracy_std': 0.0,
            'sensitivity': 0.0, 'sensitivity_std': 0.0,
            'specificity': 0.0, 'specificity_std': 0.0,
            'precision': 0.0, 'precision_std': 0.0,
            'f1_score': 0.0, 'f1_score_std': 0.0,
            'auc': 0.0, 'auc_std': 0.0,
            'fold_metrics': {},
            'n_valid_folds': 0,
            'fitness_split': 'validation',
            'fitness_metric': fitness_metric,
            'selection_metric': selection_metric,
        }
        return 0.0, None, empty_metrics


def train_fold_in_thread(genome: dict, fold_num: int, config: dict, device: torch.device) -> Tuple[int, float, nn.Module, dict]:
    """
    Entrena un modelo en un fold especifico (para ejecutar en un thread).

    Args:
        genome: Genome dictionary defining the architecture
        fold_num: Fold number (1-5)
        config: Configuration dictionary
        device: PyTorch device (CPU or CUDA)

    Returns:
        Tuple of (fold_num, score, model, metrics)
    """
    try:
        fold_train_loader, fold_validation_loader = load_fold_data(
            fold_num,
            config,
            device,
            eval_split='validation',
        )

        try:
            model = EvolvableCNN(genome, config).to(device)
        except ValueError as e:
            if "Invalid architecture" in str(e) or "Expected more than 1 value per channel" in str(e):
                print(f"      x Fold {fold_num}: Invalid architecture - {str(e)[:120]}")
                return fold_num, 0.0, None, None
            raise

        optimizer_class = OPTIMIZERS[genome['optimizer']]
        optimizer = optimizer_class(model.parameters(), lr=genome['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        fitness_metric = resolve_metric_name(config, 'fitness_metric')
        selection_metric = resolve_metric_name(config, 'fold_selection_metric')
        best_selection_score = float('-inf')
        best_selection_metrics = None
        best_epoch = 0
        best_state = None
        patience_left = int(config.get('epoch_patience', 10))
        max_epochs = int(config.get('num_epochs', 30))
        improvement_threshold = resolve_metric_improvement_threshold(config)
        validation_frequency = max(1, int(config.get('validation_frequency_epochs', 2)))

        amp_enabled = bool(config.get('use_amp', True)) and device.type == 'cuda'
        amp_dtype_name = str(config.get('amp_dtype', 'float16')).lower()
        amp_dtype = torch.bfloat16 if amp_dtype_name == 'bfloat16' else torch.float16
        autocast_device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        for epoch_idx in range(max_epochs):
            model.train()
            batch_count = 0
            max_batches = min(len(fold_train_loader), int(config.get('early_stopping_patience', len(fold_train_loader))))

            for data, target in fold_train_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
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

            model.eval()
            validation_targets = []
            validation_preds = []
            validation_probs = []
            with torch.no_grad():
                for data, target in fold_validation_loader:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    with torch.autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=amp_enabled):
                        output = model(data)
                    probs = F.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)

                    validation_targets.extend(target.cpu().numpy().tolist())
                    validation_preds.extend(predicted.cpu().numpy().tolist())
                    validation_probs.extend(probs[:, 1].cpu().numpy().tolist())

            validation_metrics = compute_classification_metrics(
                validation_targets,
                validation_preds,
                validation_probs,
            )
            current_selection_score = metric_value(validation_metrics, selection_metric)

            if current_selection_score > (best_selection_score + improvement_threshold):
                best_selection_score = current_selection_score
                best_selection_metrics = validation_metrics
                best_epoch = epoch_idx + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_left = int(config.get('epoch_patience', 10))
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, target in fold_validation_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=amp_enabled):
                    output = model(data)
                probs = F.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)

                all_targets.extend(target.cpu().numpy().tolist())
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_probs.extend(probs[:, 1].cpu().numpy().tolist())

        metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
        if best_selection_metrics is None:
            best_selection_metrics = metrics
            best_selection_score = metric_value(best_selection_metrics, selection_metric)

        metrics.update({
            'evaluation_split': 'validation',
            'fitness_metric': fitness_metric,
            'selection_metric': selection_metric,
            'best_selection_score': float(best_selection_score),
            'best_epoch': best_epoch,
            'best_selection_metrics': best_selection_metrics,
        })

        print(
            f"      -> Fold {fold_num} validation completed: "
            f"selected_by={selection_metric} best={metrics['best_selection_score']:.2f}% "
            f"epoch={best_epoch}, "
            f"Acc={metrics['accuracy']:.2f}%, Sen={metrics['sensitivity']:.2f}%, "
            f"Spe={metrics['specificity']:.2f}%, F1={metrics['f1_score']:.2f}%, "
            f"AUC={metrics['auc']:.2f}%"
        )

        return fold_num, metric_value(metrics, fitness_metric), model, metrics

    except Exception as e:
        print(f"      ERROR in Fold {fold_num}: {e}")
        import traceback
        traceback.print_exc()
        return fold_num, 0.0, None, None


def _reshape_features_if_needed(x_values: np.ndarray) -> np.ndarray:
    """Ensures 2D sequence arrays include a channel dimension."""
    if len(x_values.shape) == 2:
        return x_values.reshape((x_values.shape[0], 1, x_values.shape[1]))
    return x_values


def _load_split_dataset(
    fold_files_directory: str,
    dataset_id: str,
    fold_number: int,
    split_name: str,
    cache_mode: str,
) -> torch.utils.data.TensorDataset:
    """Loads one fold split and converts it to a TensorDataset."""
    x_values = _load_numpy_array(
        os.path.join(fold_files_directory, f'X_{split_name}_{dataset_id}_fold_{fold_number}.npy'),
        cache_mode,
    )
    y_values = _load_numpy_array(
        os.path.join(fold_files_directory, f'y_{split_name}_{dataset_id}_fold_{fold_number}.npy'),
        cache_mode,
    )

    x_tensor = torch.tensor(_reshape_features_if_needed(x_values), dtype=torch.float32)
    y_tensor = torch.tensor(y_values, dtype=torch.long)
    return torch.utils.data.TensorDataset(x_tensor, y_tensor)


def _build_dataloader(
    dataset: torch.utils.data.TensorDataset,
    config: dict,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    """Creates a DataLoader using the configured worker settings."""
    num_workers, persistent_workers, prefetch_factor, pin_memory = _resolve_dataloader_settings(config, device)
    loader_kwargs = {
        'batch_size': config['batch_size'],
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    return DataLoader(dataset, shuffle=shuffle, **loader_kwargs)


def load_fold_data(
    fold_number: int,
    config: dict,
    device: torch.device,
    eval_split: str = 'validation',
) -> Tuple[DataLoader, ...]:
    """
    Carga los datos de un fold especifico para el entrenamiento.

    Args:
        fold_number: Numero de fold (1-5)
        config: Configuration dictionary
        device: PyTorch device (CPU or CUDA)
        eval_split: Which evaluation split to load. Use "validation" for
            evolutionary fitness, "test" for final reporting,
            "validation_and_test" only for explicit legacy compatibility, or
            "all" to return train, validation, and test loaders separately.

    Returns:
        Tuple de DataLoaders. Returns (train, eval) for split-specific modes
        or (train, validation, test) for eval_split="all".
    """
    eval_split = _normalize_eval_split(eval_split)
    cache_mode = _resolve_cache_mode(config)
    cache_enabled = cache_mode in {'ram', 'memmap'}

    if not cache_enabled:
        return _load_fold_data_uncached(fold_number, config, device, cache_mode, eval_split)

    cache_key = _build_fold_cache_key(fold_number, config, device, cache_mode, eval_split)
    with _FOLD_DATALOADER_CACHE_LOCK:
        cached = _FOLD_DATALOADER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    loaded = _load_fold_data_uncached(fold_number, config, device, cache_mode, eval_split)
    with _FOLD_DATALOADER_CACHE_LOCK:
        # Avoid duplicate work if another thread loaded the same key in parallel.
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
    eval_split: str,
) -> Tuple[DataLoader, ...]:
    """Loads fold data and creates DataLoaders without cache lookup."""
    eval_split = _normalize_eval_split(eval_split)
    fold_files_directory = _resolve_fold_files_directory(config)
    dataset_id = config['dataset_id']

    train_dataset = _load_split_dataset(
        fold_files_directory,
        dataset_id,
        fold_number,
        'train',
        cache_mode,
    )
    fold_train_loader = _build_dataloader(train_dataset, config, device, shuffle=True)

    if eval_split == 'all':
        validation_dataset = _load_split_dataset(
            fold_files_directory,
            dataset_id,
            fold_number,
            'val',
            cache_mode,
        )
        test_dataset = _load_split_dataset(
            fold_files_directory,
            dataset_id,
            fold_number,
            'test',
            cache_mode,
        )
        return (
            fold_train_loader,
            _build_dataloader(validation_dataset, config, device, shuffle=False),
            _build_dataloader(test_dataset, config, device, shuffle=False),
        )

    if eval_split == 'validation':
        eval_dataset = _load_split_dataset(
            fold_files_directory,
            dataset_id,
            fold_number,
            'val',
            cache_mode,
        )
    elif eval_split == 'test':
        eval_dataset = _load_split_dataset(
            fold_files_directory,
            dataset_id,
            fold_number,
            'test',
            cache_mode,
        )
    else:
        validation_dataset = _load_split_dataset(
            fold_files_directory,
            dataset_id,
            fold_number,
            'val',
            cache_mode,
        )
        test_dataset = _load_split_dataset(
            fold_files_directory,
            dataset_id,
            fold_number,
            'test',
            cache_mode,
        )
        x_eval = torch.cat([validation_dataset.tensors[0], test_dataset.tensors[0]], dim=0)
        y_eval = torch.cat([validation_dataset.tensors[1], test_dataset.tensors[1]], dim=0)
        eval_dataset = torch.utils.data.TensorDataset(x_eval, y_eval)

    return fold_train_loader, _build_dataloader(eval_dataset, config, device, shuffle=False)
