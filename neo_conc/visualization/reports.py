"""
Reporting utilities for notebook orchestration.

Extracts verbose reporting logic from notebooks while preserving output format.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import torch

from ..models.evolvable_cnn import EvolvableCNN


def display_best_architecture(
    best_genome: dict,
    config: dict,
    neuroevolution,
    execution_time: Optional[object] = None
) -> None:
    """
    Display detailed best-architecture report and save JSON snapshot.

    Args:
        best_genome: Best genome dictionary.
        config: Configuration dictionary.
        neuroevolution: HybridNeuroevolution instance.
        execution_time: Optional execution time metadata.
    """
    print("=" * 60)
    print("    BEST EVOLVED ARCHITECTURE (1D AUDIO)")
    print("=" * 60)

    print("\nGENERAL INFORMATION:")
    print(f"   Genome ID: {best_genome['id']}")
    print(f"   Fitness Achieved: {best_genome['fitness']:.2f}%")
    print(f"   Generation: {neuroevolution.generation}")
    print(f"   Dataset: {config['dataset']}")
    print(f"   Dataset ID: {config.get('dataset_id', 'N/A')}")
    print(f"   Fold: {config.get('current_fold', 'N/A')}")

    print("\nNETWORK ARCHITECTURE:")
    print(f"   Input: 1D Audio Signal (length={config['sequence_length']})")
    print(f"   Convolutional Layers (Conv1D): {best_genome['num_conv_layers']}")
    print(f"   Fully Connected Layers: {best_genome['num_fc_layers']}")
    print(f"   Output: {config['num_classes']} classes")

    print("\nCONVOLUTIONAL LAYER DETAILS (1D):")
    for i in range(best_genome["num_conv_layers"]):
        filters = best_genome["filters"][i]
        kernel = best_genome["kernel_sizes"][i]
        activation = best_genome["activations"][i % len(best_genome["activations"])]
        print(f"   Conv1D-{i+1}: {filters} filters, kernel_size={kernel}, activation={activation}")
        print(f"             -> BatchNorm1D -> {activation.upper()} -> MaxPool1D(2)")

    print("\nFULLY CONNECTED LAYER DETAILS:")
    for i, nodes in enumerate(best_genome["fc_nodes"]):
        print(f"   FC{i+1}: {nodes} neurons -> BatchNorm1D -> ReLU -> Dropout({best_genome['dropout_rate']:.3f})")
    print(f"   Output: {config['num_classes']} neurons (Control vs Pathological)")

    print("\nHYPERPARAMETERS:")
    print(f"   Optimizer: {best_genome['optimizer'].upper()}")
    print(f"   Learning Rate: {best_genome['learning_rate']:.6f}")
    print(f"   Dropout Rate: {best_genome['dropout_rate']:.3f}")
    print(f"   Activation Functions: {', '.join(set(best_genome['activations']))}")

    print("\nCREATING FINAL MODEL...")
    try:
        final_model = EvolvableCNN(best_genome, config)
        total_params = sum(p.numel() for p in final_model.parameters())
        trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)

        print("   Model created successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")

        print("\nCOMPACT SUMMARY:")
        print(f"   {final_model.get_architecture_summary()}")

    except Exception as e:
        print(f"   ERROR creating model: {e}")

    print("\nSUMMARY TABLE:")
    print(f"{'='*80}")
    print(f"{'Parameter':<25} {'Value':<30} {'Description':<25}")
    print(f"{'='*80}")
    print(f"{'ID':<25} {best_genome['id']:<30} {'Unique identifier':<25}")
    print(f"{'Fitness':<25} {best_genome['fitness']:.2f}%{'':<25} {'Accuracy achieved':<25}")
    print(f"{'Architecture':<25} {'Conv1D + FC':<30} {'1D Convolutional':<25}")
    print(f"{'Conv Layers':<25} {best_genome['num_conv_layers']:<30} {'Conv1D layers':<25}")
    print(f"{'FC Layers':<25} {best_genome['num_fc_layers']:<30} {'FC layers':<25}")
    print(f"{'Optimizer':<25} {best_genome['optimizer']:<30} {'Optimization algorithm':<25}")
    print(f"{'Learning Rate':<25} {best_genome['learning_rate']:<30.6f} {'Learning rate':<25}")
    print(f"{'Dropout':<25} {best_genome['dropout_rate']:<30} {'Dropout rate':<25}")
    print(f"{'Input Length':<25} {config['sequence_length']:<30} {'Audio sequence length':<25}")
    print(f"{'Classes':<25} {config['num_classes']:<30} {'Binary classification':<25}")
    print(f"{'='*80}")

    print("\nCOMPARISON WITH OBJECTIVES:")
    if best_genome["fitness"] >= config["fitness_threshold"]:
        print(f"   ✓ TARGET REACHED: {best_genome['fitness']:.2f}% >= {config['fitness_threshold']}%")
    else:
        print(f"   ✗ TARGET NOT REACHED: {best_genome['fitness']:.2f}% < {config['fitness_threshold']}%")
        print(f"     Gap: {config['fitness_threshold'] - best_genome['fitness']:.2f}%")

    print(f"   Generations used: {neuroevolution.generation}/{config['max_generations']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"best_architecture_audio_{timestamp}.json"

    results_data = {
        "timestamp": timestamp,
        "execution_time": str(execution_time),
        "dataset_type": "audio_1D",
        "dataset_id": config.get("dataset_id", "N/A"),
        "fold": config.get("current_fold", "N/A"),
        "config_used": {k: v for k, v in config.items() if not k.startswith("_")},
        "best_genome": best_genome,
        "final_generation": neuroevolution.generation,
        "evolution_stats": neuroevolution.generation_stats,
    }

    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {results_file}")
    except Exception as e:
        print(f"\n✗ WARNING: Error saving results: {e}")

    print(f"\n{'='*60}")
    print("HYBRID NEUROEVOLUTION FOR AUDIO COMPLETED!")
    print(f"{'='*60}")


def print_checkpoint_info(neuroevolution, device: torch.device) -> None:
    """
    Print summary information for the currently saved best checkpoint.

    Args:
        neuroevolution: HybridNeuroevolution instance.
        device: Device used to load checkpoint metadata.
    """
    print("=" * 80)
    print("INFORMACIÓN DEL CHECKPOINT DEL MEJOR MODELO")
    print("=" * 80)

    if neuroevolution.best_checkpoint_path:
        print(f"\n✓ Checkpoint guardado en: {neuroevolution.best_checkpoint_path}")

        if os.path.exists(neuroevolution.best_checkpoint_path):
            file_size = os.path.getsize(neuroevolution.best_checkpoint_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"  Tamaño: {file_size_mb:.2f} MB")

            checkpoint_data = torch.load(neuroevolution.best_checkpoint_path, map_location=device, weights_only=False)
            print("\n  Información del modelo guardado:")
            print(f"    Generación: {checkpoint_data['generation']}")
            print(f"    Fitness: {checkpoint_data['fitness']:.2f}%")
            print(f"    ID Genoma: {checkpoint_data['genome']['id']}")
            print(
                f"    Arquitectura: {checkpoint_data['genome']['num_conv_layers']} Conv1D + "
                f"{checkpoint_data['genome']['num_fc_layers']} FC"
            )
            print(f"    Optimizador: {checkpoint_data['genome']['optimizer']}")
            print(f"    Learning Rate: {checkpoint_data['genome']['learning_rate']}")

            print("\n  Este checkpoint se usará como punto de partida para el 5-fold CV")
            print("  (Transfer learning desde el modelo pre-entrenado)")
        else:
            print("  ✗ Archivo no encontrado")
    else:
        print("\n✗ No hay checkpoint disponible")
        print("  El 5-fold CV entrenará desde cero")

    print("\n" + "=" * 80)
