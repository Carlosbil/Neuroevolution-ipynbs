"""Synthetic-to-real final evaluation utilities for the best genome.

Models are fitted and selected exclusively with synthetic train/validation
data, reported on synthetic test, and finally evaluated on real held-out test
data without any real-data weight updates.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from ..config import OPTIMIZERS, get_final_evaluation_config
from ..evolution.fitness import (
    FoldLoaders,
    load_fold_loaders as load_fold_loaders_from_evolution,
    load_fold_test_loader as load_fold_test_loader_from_evolution,
)
from ..models.evolvable_cnn import EvolvableCNN


def _to_json_compatible(value: Any) -> Any:
    """Recursively convert NumPy and tensor values to JSON-native types."""
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def load_fold_loaders(
    config: dict,
    fold_number: int,
    device: Optional[torch.device] = None
) -> FoldLoaders:
    """
    Load train/validation/test dataloaders for a specific fold.

    Args:
        config: System configuration dictionary.
        fold_number: Fold number (1-5).
        device: PyTorch device. If not provided, auto-detected.

    Returns:
        FoldLoaders with train, validation, and test loaders.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_fold_loaders_from_evolution(fold_number, config, device)


def load_fold_data(
    config: dict,
    fold_number: int,
    device: Optional[torch.device] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Compatibility wrapper returning train and held-out test loaders.

    New code should use load_fold_loaders() to keep validation explicit.
    """
    loaders = load_fold_loaders(config, fold_number, device)
    return loaders.train, loaders.test


def load_fold_test_loader(
    config: dict,
    fold_number: int,
    device: Optional[torch.device] = None,
) -> torch.utils.data.DataLoader:
    """Load only the test split for a source and fold."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_fold_test_loader_from_evolution(fold_number, config, device)


def _evaluate_loader_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate a model on one loader and return classification metrics."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)

            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_probs = np.array(all_probs)

    accuracy = accuracy_score(y_true, y_pred) * 100
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100

    try:
        auc = roc_auc_score(y_true, y_probs[:, 1]) * 100
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "n_samples": len(y_true),
    }


def evaluate_single_fold(
    best_genome: dict,
    config: dict,
    fold_train_loader: torch.utils.data.DataLoader,
    fold_validation_loader: torch.utils.data.DataLoader,
    fold_test_loader: torch.utils.data.DataLoader,
    fold_num: int,
    device: torch.device,
    num_epochs: int = 100,
    fold_real_test_loader: Optional[torch.utils.data.DataLoader] = None,
) -> Dict[str, Any]:
    """
    Fit on synthetic data, report synthetic test and evaluate on real test.

    Args:
        best_genome: Best architecture genome.
        config: System configuration dictionary.
        fold_train_loader: Synthetic training dataloader.
        fold_validation_loader: Synthetic validation dataloader used for selection.
        fold_test_loader: Synthetic test dataloader used for internal reporting.
        fold_num: Fold number (1-5).
        device: Device to train/evaluate on.
        num_epochs: Max epochs for this fold.
        fold_real_test_loader: Real held-out test loader. When omitted, the
            synthetic test loader is used for backward compatibility.

    Returns:
        Dictionary with fold metrics and metadata.
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num}/5")
    print(f"{'='*70}")

    model = EvolvableCNN(best_genome, config).to(device)

    optimizer_class = OPTIMIZERS[best_genome["optimizer"]]
    optimizer = optimizer_class(model.parameters(), lr=best_genome["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    checkpoint_metric = config.get("checkpoint_metric", config.get("fitness_metric", "f1_score"))
    best_validation_score = -float("inf")
    best_model_state = None
    best_epoch = 0

    patience = config.get("epoch_patience", 10)
    patience_counter = 0
    last_improvement_score = 0.0
    improvement_threshold = config.get("improvement_threshold", 0.01)

    print(f"Entrenando SOLO con datos sintéticos por hasta {num_epochs} épocas (patience={patience})...")
    print(f"Guardando el MEJOR modelo basado en validation sintética ({checkpoint_metric})")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for data, target in fold_train_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / max(1, batch_count)

        validation_metrics = _evaluate_loader_metrics(model, fold_validation_loader, device)
        current_score = validation_metrics.get(checkpoint_metric, validation_metrics["f1_score"])

        if current_score > best_validation_score:
            best_validation_score = current_score
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            print(
                f"   Época {epoch}/{num_epochs}: loss={avg_loss:.4f}, "
                f"val_{checkpoint_metric}={current_score:.2f}% *** NUEVO MEJOR ***"
            )

        improvement = current_score - last_improvement_score
        if improvement >= improvement_threshold:
            patience_counter = 0
            last_improvement_score = current_score
        else:
            patience_counter += 1

        if epoch % 30 == 0 or epoch == 1:
            print(
                f"   Época {epoch}/{num_epochs}: loss={avg_loss:.4f}, "
                f"val_{checkpoint_metric}={current_score:.2f}% "
                f"(best={best_validation_score:.2f}%)"
            )

        if patience_counter >= patience:
            print(f"   Early stopping en época {epoch} (sin mejora por {patience} épocas)")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"\n   ✓ Cargado mejor modelo de época {best_epoch} "
            f"(validation {checkpoint_metric}={best_validation_score:.2f}%)"
        )
    else:
        print("\n   ⚠ Usando modelo final (no se encontró mejora)")

    print("Evaluando con el mejor modelo sobre test sintético (reporting interno)...")
    synthetic_test_metrics = _evaluate_loader_metrics(model, fold_test_loader, device)

    final_loader = fold_real_test_loader if fold_real_test_loader is not None else fold_test_loader
    final_data_source = "real" if fold_real_test_loader is not None else "synthetic"
    final_evaluation_split = "real_test" if fold_real_test_loader is not None else "synthetic_test"
    print(f"Evaluando SIN REENTRENAR sobre {final_evaluation_split}...")
    final_metrics = _evaluate_loader_metrics(model, final_loader, device)

    print(f"\nResultados finales Fold {fold_num} sobre {final_evaluation_split} (mejor época sintética {best_epoch}):")
    print(f"   Accuracy:     {final_metrics['accuracy']:.2f}%")
    print(f"   Sensitivity:  {final_metrics['sensitivity']:.2f}%")
    print(f"   Specificity:  {final_metrics['specificity']:.2f}%")
    print(f"   F1-Score:     {final_metrics['f1_score']:.2f}%")
    print(f"   AUC:          {final_metrics['auc']:.2f}%")

    return {
        "fold": fold_num,
        "accuracy": final_metrics["accuracy"],
        "sensitivity": final_metrics["sensitivity"],
        "specificity": final_metrics["specificity"],
        "f1_score": final_metrics["f1_score"],
        "auc": final_metrics["auc"],
        "confusion_matrix": final_metrics["confusion_matrix"],
        "n_samples": final_metrics["n_samples"],
        "best_epoch": best_epoch,
        "training_data_source": "synthetic",
        "selection_data_source": "synthetic",
        "internal_test_data_source": "synthetic",
        "final_evaluation_data_source": final_data_source,
        "selection_split": "synthetic_validation",
        "internal_evaluation_split": "synthetic_test",
        "evaluation_split": final_evaluation_split,
        "synthetic_test_metrics": synthetic_test_metrics,
        "checkpoint_metric": checkpoint_metric,
        "best_validation_score": best_validation_score,
    }


def evaluate_5fold_cross_validation(
    best_genome: dict,
    config: dict,
    device: torch.device,
    num_epochs: Optional[int] = None,
    neuroevolution_instance=None
) -> Optional[Dict[str, Any]]:
    """
    Evaluate synthetic-to-real generalization over five paired partitions.

    Args:
        best_genome: Best architecture genome.
        config: System configuration dictionary.
        device: Device to train/evaluate on.
        num_epochs: Optional epochs per fold. If None, uses config['num_epochs'].
        neuroevolution_instance: Optional HybridNeuroevolution instance to load checkpoint.

    Returns:
        Aggregated real held-out test results or None if all folds fail.
    """
    if num_epochs is None:
        num_epochs = config.get("num_epochs", 100)

    print("=" * 80)
    print("EVALUACIÓN FINAL 5-FOLD: ENTRENAMIENTO SINTÉTICO -> TEST REAL")
    print("=" * 80)

    print("\nPROTOCOLO DE GENERALIZACIÓN ENTRE DOMINIOS:")
    print(f"   - Entrena por {num_epochs} épocas exclusivamente con train sintético")
    checkpoint_metric = config.get("checkpoint_metric", config.get("fitness_metric", "f1_score"))
    print(f"   - Selecciona el MEJOR modelo con {checkpoint_metric} de validation sintética")
    print(f"   - Aplica early stopping con patience={config.get('epoch_patience', 10)}")
    print("   - Reporta test sintético sin usarlo para seleccionar")
    print("   - Evalúa finalmente sobre test real SIN reentrenar con datos reales")

    print("\nArquitectura a evaluar:")
    print(f"   Conv1D Layers: {best_genome['num_conv_layers']}")
    print(f"   FC Layers: {best_genome['num_fc_layers']}")
    print(f"   Optimizer: {best_genome['optimizer']}")
    print(f"   Learning Rate: {best_genome['learning_rate']}")
    print(f"   Épocas por fold: {num_epochs}")

    if neuroevolution_instance is not None:
        print("\nLa evaluación final usa la arquitectura seleccionada y la reentrena desde cero con sintéticos por fold.")
        print("Los pesos del checkpoint evolutivo no se reutilizan y ningún dato real actualiza pesos.")
    else:
        print("\nEntrenando la arquitectura desde cero con datos sintéticos por fold")

    fold_results = []
    synthetic_config = dict(config)
    real_config = get_final_evaluation_config(config)
    synthetic_identity = (
        os.path.abspath(synthetic_config['data_path']),
        synthetic_config['fold_files_subdirectory'],
        synthetic_config['dataset_id'],
    )
    real_identity = (
        os.path.abspath(real_config['data_path']),
        real_config['fold_files_subdirectory'],
        real_config['dataset_id'],
    )
    if synthetic_identity == real_identity:
        raise ValueError(
            "Final evaluation must use a distinct real dataset source; "
            "synthetic and real configurations currently resolve to the same files."
        )

    for fold_num in range(1, 6):
        print(f"\n\nCargando fuentes sintética y real del Fold {fold_num}...")

        try:
            synthetic_loaders = load_fold_loaders(synthetic_config, fold_num, device=device)
            real_test_loader = load_fold_test_loader(real_config, fold_num, device=device)
            print(f"   Synthetic train batches: {len(synthetic_loaders.train)}")
            print(f"   Synthetic validation batches: {len(synthetic_loaders.validation)}")
            print(f"   Synthetic test batches: {len(synthetic_loaders.test)}")
            print(f"   Real final-test batches: {len(real_test_loader)}")

            fold_result = evaluate_single_fold(
                best_genome,
                synthetic_config,
                synthetic_loaders.train,
                synthetic_loaders.validation,
                synthetic_loaders.test,
                fold_num,
                device=device,
                num_epochs=num_epochs,
                fold_real_test_loader=real_test_loader,
            )
            fold_results.append(fold_result)

        except Exception as e:
            print(f"   ERROR en Fold {fold_num}: {e}")
            print("   Saltando este fold...")
            import traceback

            traceback.print_exc()
            continue

    print("\n\n" + "=" * 80)
    print("RESULTADOS AGREGADOS (5-FOLD TEST REAL, SIN ENTRENAMIENTO REAL)")
    print("=" * 80)

    if not fold_results:
        print("ERROR: No se pudo evaluar ningún fold")
        return None

    accuracies = [r["accuracy"] for r in fold_results]
    sensitivities = [r["sensitivity"] for r in fold_results]
    specificities = [r["specificity"] for r in fold_results]
    f1_scores = [r["f1_score"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    mean_sensitivity = np.mean(sensitivities)
    std_sensitivity = np.std(sensitivities)

    mean_specificity = np.mean(specificities)
    std_specificity = np.std(specificities)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    synthetic_test_results = [
        result["synthetic_test_metrics"]
        for result in fold_results
        if result.get("synthetic_test_metrics")
    ]
    synthetic_test_summary = None
    if synthetic_test_results:
        synthetic_test_summary = {
            "mean_accuracy": np.mean([m["accuracy"] for m in synthetic_test_results]),
            "std_accuracy": np.std([m["accuracy"] for m in synthetic_test_results]),
            "mean_sensitivity": np.mean([m["sensitivity"] for m in synthetic_test_results]),
            "std_sensitivity": np.std([m["sensitivity"] for m in synthetic_test_results]),
            "mean_specificity": np.mean([m["specificity"] for m in synthetic_test_results]),
            "std_specificity": np.std([m["specificity"] for m in synthetic_test_results]),
            "mean_f1": np.mean([m["f1_score"] for m in synthetic_test_results]),
            "std_f1": np.std([m["f1_score"] for m in synthetic_test_results]),
            "mean_auc": np.mean([m["auc"] for m in synthetic_test_results]),
            "std_auc": np.std([m["auc"] for m in synthetic_test_results]),
        }

    print("\nRESULTADOS POR FOLD:")
    print(f"{'Fold':<6} {'Accuracy':<12} {'Sensitivity':<14} {'Specificity':<14} {'F1-Score':<12} {'AUC':<12} {'Best Epoch':<12}")
    print("-" * 95)
    for r in fold_results:
        best_ep = r.get("best_epoch", "N/A")
        print(
            f"{r['fold']:<6} {r['accuracy']:>6.2f}%      {r['sensitivity']:>6.2f}%        "
            f"{r['specificity']:>6.2f}%        {r['f1_score']:>6.2f}%      {r['auc']:>6.2f}%      {best_ep}"
        )

    print("-" * 95)
    print(f"{'Mean':<6} {mean_accuracy:>6.2f}%      {mean_sensitivity:>6.2f}%        {mean_specificity:>6.2f}%        {mean_f1:>6.2f}%      {mean_auc:>6.2f}%")
    print(f"{'Std':<6} {std_accuracy:>6.2f}%      {std_sensitivity:>6.2f}%        {std_specificity:>6.2f}%        {std_f1:>6.2f}%      {std_auc:>6.2f}%")

    results = {
        "fold_results": fold_results,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_sensitivity": mean_sensitivity,
        "std_sensitivity": std_sensitivity,
        "mean_specificity": mean_specificity,
        "std_specificity": std_specificity,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "n_folds": len(fold_results),
        "architecture": f"{best_genome['num_conv_layers']}Conv1D+{best_genome['num_fc_layers']}FC",
        "num_epochs_used": num_epochs,
        "training_data_source": "synthetic",
        "selection_data_source": "synthetic",
        "internal_test_data_source": "synthetic",
        "final_evaluation_data_source": "real",
        "training_dataset_id": synthetic_config.get("dataset_id"),
        "training_fold_files_subdirectory": synthetic_config.get("fold_files_subdirectory"),
        "final_evaluation_dataset_id": real_config.get("dataset_id"),
        "final_evaluation_fold_files_subdirectory": real_config.get("fold_files_subdirectory"),
        "selection_split": "synthetic_validation",
        "internal_evaluation_split": "synthetic_test",
        "evaluation_split": "real_test",
        "real_data_used_for_weight_updates": False,
        "synthetic_test_summary": synthetic_test_summary,
    }

    print("\n" + "=" * 80)
    print("FORMATO PARA TABLA")
    print("=" * 80)

    print("\nMÉTRICAS FINALES (promedio ± desviación estándar):")
    print(f"   Accuracy:     {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"   Sensitivity:  {mean_sensitivity:.2f}% ± {std_sensitivity:.2f}%")
    print(f"   Specificity:  {mean_specificity:.2f}% ± {std_specificity:.2f}%")
    print(f"   F1-Score:     {mean_f1:.2f}% ± {std_f1:.2f}%")
    print(f"   AUC:          {mean_auc:.2f}% ± {std_auc:.2f}%")

    print("\nFORMATO PARA TABLA (valores en escala 0-1):")
    print(f"   Model: Neuroevolution-{results['architecture']}")
    print(f"   Accuracy:     {mean_accuracy/100:.2f} ({int(std_accuracy)}%)")
    print(f"   Sensitivity:  {mean_sensitivity/100:.2f} ({int(std_sensitivity)}%)")
    print(f"   Specificity:  {mean_specificity/100:.2f} ({int(std_specificity)}%)")
    print(f"   F1-Score:     {mean_f1/100:.2f} ({int(std_f1)}%)")
    print(f"   AUC:          {mean_auc/100:.2f} ({int(std_auc)}%)")

    print("\nFORMATO LaTeX:")
    latex_row = (
        f"Neuroevolution-{results['architecture']} & {mean_accuracy/100:.2f} ({int(std_accuracy)}\\%) & "
        f"{mean_sensitivity/100:.2f} ({int(std_sensitivity)}\\%) & {mean_specificity/100:.2f} ({int(std_specificity)}\\%) & "
        f"{mean_f1/100:.2f} ({int(std_f1)}\\%) & {mean_auc/100:.2f} ({int(std_auc)}\\%) \\\\"
    )
    print(f"   {latex_row}")

    print("\nFORMATO Markdown:")
    markdown_row = (
        f"| Neuroevolution-{results['architecture']} | {mean_accuracy/100:.2f} ({int(std_accuracy)}%) | "
        f"{mean_sensitivity/100:.2f} ({int(std_sensitivity)}%) | {mean_specificity/100:.2f} ({int(std_specificity)}%) | "
        f"{mean_f1/100:.2f} ({int(std_f1)}%) | {mean_auc/100:.2f} ({int(std_auc)}%) |"
    )
    print(f"   {markdown_row}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = config.get("artifacts_dir", "artifacts/test_audio")
    os.makedirs(artifacts_dir, exist_ok=True)
    results_file = os.path.join(artifacts_dir, f"synthetic_to_real_5fold_results_{timestamp}.json")
    results["results_path"] = results_file

    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(_to_json_compatible(results), f, indent=2)
        print(f"\n✓ Resultados guardados en: {results_file}")
    except Exception as e:
        print(f"\n✗ Error guardando resultados: {e}")

    print("\n" + "=" * 80)

    return results
