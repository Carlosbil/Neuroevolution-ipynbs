"""
Fitness evaluation module for parallel 5-fold cross-validation.

This module implements parallel training and evaluation of genomes using ThreadPoolExecutor
to run all 5 folds simultaneously. Fitness is calculated as the average F1-score across folds.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict

from ..models.evolvable_cnn import EvolvableCNN
from ..config import OPTIMIZERS


def evaluate_fitness(genome: dict, config: dict, device: torch.device) -> Tuple[float, nn.Module, dict]:
    """
    Evalua el fitness de un genoma usando 5-fold cross-validation PARALELO.
    Los 5 folds se entrenan en threads separados y se espera a que terminen todos.
    El fitness final es el promedio de F1-score de los 5 folds.

    Args:
        genome: Genome dictionary defining the architecture
        config: Configuration dictionary
        device: PyTorch device (CPU or CUDA)

    Returns:
        Tuple de (fitness, model, metrics) donde:
            - fitness: promedio de F1-score de los 5 folds
            - model: modelo entrenado en el mejor fold (para checkpoint)
            - metrics: diccionario con metricas agregadas de todos los folds
    """
    print(f"      Training/Evaluating model {genome['id']} with PARALLEL 5-FOLD CROSS-VALIDATION")

    fold_accuracies = {}
    fold_models = {}
    fold_metrics = {}

    try:
        # Usar ThreadPoolExecutor para ejecutar los 5 folds en paralelo
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Enviar los 5 folds a threads separados
            print("      -> Submitting 5 folds to thread pool...")
            futures = {
                executor.submit(train_fold_in_thread, genome, fold_num, config, device): fold_num
                for fold_num in range(1, 6)
            }

            # Esperar a que todos los folds terminen
            print("      -> Waiting for all 5 folds to complete...")
            for future in as_completed(futures):
                fold_num, fold_score, model, metrics = future.result()
                fold_accuracies[fold_num] = fold_score
                fold_models[fold_num] = model
                fold_metrics[fold_num] = metrics

        # Ordenar resultados por fold_num
        sorted_folds = sorted(fold_accuracies.keys())
        f1_scores_list = [fold_accuracies[f] for f in sorted_folds]

        # Encontrar el mejor modelo
        best_fold_num = max(fold_accuracies, key=fold_accuracies.get)
        best_fold_f1 = fold_accuracies[best_fold_num]
        best_model = fold_models[best_fold_num]

        # Calcular fitness como promedio de los 5 folds
        avg_fitness = np.mean(f1_scores_list)
        std_fitness = np.std(f1_scores_list)

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
                'n_valid_folds': len(valid_metrics)
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
                'n_valid_folds': 0
            }

        print(f"      + PARALLEL 5-Fold CV Results for {genome['id']}:")
        print(f"        Fold F1-scores: {[f'{score:.2f}%' for score in f1_scores_list]}")
        print(f"        Average fitness: {avg_fitness:.2f}% +/- {std_fitness:.2f}%")
        print(f"        Best fold: Fold {best_fold_num} with {best_fold_f1:.2f}% F1")
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
            'n_valid_folds': 0
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
        fold_train_loader, fold_test_loader = load_fold_data(fold_num, config, device)

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

        best_acc = 0.0
        best_state = None
        patience_left = int(config.get('epoch_patience', 10))
        max_epochs = int(config.get('num_epochs', 30))
        improvement_threshold = float(config.get('improvement_threshold', 0.01))

        for _ in range(max_epochs):
            model.train()
            batch_count = 0
            max_batches = min(len(fold_train_loader), int(config.get('early_stopping_patience', len(fold_train_loader))))

            for data, target in fold_train_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                batch_count += 1
                if batch_count >= max_batches:
                    break

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in fold_test_loader:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
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

        model.eval()
        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, target in fold_test_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
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
            'auc': float(auc)
        }

        print(
            f"      -> Fold {fold_num} completed: "
            f"Acc={metrics['accuracy']:.2f}%, Sen={metrics['sensitivity']:.2f}%, "
            f"Spe={metrics['specificity']:.2f}%, F1={metrics['f1_score']:.2f}%, "
            f"AUC={metrics['auc']:.2f}%"
        )

        return fold_num, metrics['f1_score'], model, metrics

    except Exception as e:
        print(f"      ERROR in Fold {fold_num}: {e}")
        import traceback
        traceback.print_exc()
        return fold_num, 0.0, None, None


def load_fold_data(fold_number: int, config: dict, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    """
    Carga los datos de un fold especifico para el entrenamiento.

    Args:
        fold_number: Numero de fold (1-5)
        config: Configuration dictionary
        device: PyTorch device (CPU or CUDA)

    Returns:
        Tuple de (train_loader, test_loader)
    """
    fold_files_directory = os.path.join(
        config['data_path'],
        f"files_real_{config['fold_id']}"
    )

    dataset_id = config['dataset_id']

    # Cargar datos del fold
    x_train = np.load(os.path.join(fold_files_directory, f'X_train_{dataset_id}_fold_{fold_number}.npy'))
    y_train = np.load(os.path.join(fold_files_directory, f'y_train_{dataset_id}_fold_{fold_number}.npy'))
    x_val = np.load(os.path.join(fold_files_directory, f'X_val_{dataset_id}_fold_{fold_number}.npy'))
    y_val = np.load(os.path.join(fold_files_directory, f'y_val_{dataset_id}_fold_{fold_number}.npy'))
    x_test = np.load(os.path.join(fold_files_directory, f'X_test_{dataset_id}_fold_{fold_number}.npy'))
    y_test = np.load(os.path.join(fold_files_directory, f'y_test_{dataset_id}_fold_{fold_number}.npy'))

    # Reshape si es necesario
    if len(x_train.shape) == 2:
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # Convertir a tensores
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(y_train.astype(np.int64))
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.LongTensor(y_val.astype(np.int64))
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.LongTensor(y_test.astype(np.int64))

    # Crear datasets
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    x_eval = torch.cat([x_val_tensor, x_test_tensor], dim=0)
    y_eval = torch.cat([y_val_tensor, y_test_tensor], dim=0)
    test_dataset = torch.utils.data.TensorDataset(x_eval, y_eval)

    # Crear DataLoaders
    fold_train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    fold_test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return fold_train_loader, fold_test_loader
