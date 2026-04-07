"""
Neuroevolution package for audio classification.

This package provides modular components for hybrid neuroevolution,
combining genetic algorithms with neural network training.
"""

__version__ = "0.1.0"

# Core configuration and utilities
from .config import (
    CONFIG,
    ACTIVATION_FUNCTIONS,
    OPTIMIZERS,
    REQUIRED_PACKAGES,
    get_default_config,
    validate_config,
)
from .device_utils import setup_device_and_seeds, get_device
from .logger import setup_notebook_logging, verify_dependencies, install_package

# Data loading
from .data.loader import load_dataset

# Model components
from .models.evolvable_cnn import EvolvableCNN
from .models.genome_validator import (
    is_genome_valid,
    validate_and_fix_genome,
    calculate_max_safe_conv_layers,
)

# Genetic operators
from .genetics.genome import create_random_genome
from .genetics.mutation import mutate_genome
from .genetics.crossover import crossover_genomes
from .genetics.selection import select_population
from .genetics.speciation import calculate_compatibility_distance, assign_species
from .genetics.innovation import (
    INNOVATION_NAMESPACE,
    innovation_uuid,
    build_innovation_genes,
    append_structural_event,
)

# Evolution engine
from .evolution.engine import HybridNeuroevolution
from .evolution.fitness import evaluate_fitness, train_fold_in_thread, load_fold_data

# Evaluation and artifacts
from .evaluation.metrics import calculate_metrics, aggregate_fold_metrics
from .evaluation.artifacts import ArtifactManager
from .evaluation.cross_validation import evaluate_5fold_cross_validation, evaluate_single_fold

# Visualization
from .visualization.plots import (
    plot_fitness_evolution,
    show_evolution_statistics,
    analyze_failed_evaluations,
    configure_plot_style,
)
from .visualization.reports import display_best_architecture, print_checkpoint_info

# Backward-compatible aliases
setup_device = setup_device_and_seeds
setup_seeds = setup_device_and_seeds
setup_logging = setup_notebook_logging
install_packages = verify_dependencies
select_parents = select_population
assign_to_species = assign_species

__all__ = [
    # Configuration
    "CONFIG",
    "ACTIVATION_FUNCTIONS",
    "OPTIMIZERS",
    "REQUIRED_PACKAGES",
    "get_default_config",
    "validate_config",
    # Device and logging
    "setup_device_and_seeds",
    "get_device",
    "setup_notebook_logging",
    "verify_dependencies",
    "install_package",
    # Backward-compatible aliases
    "setup_device",
    "setup_seeds",
    "setup_logging",
    "install_packages",
    # Data
    "load_dataset",
    # Models
    "EvolvableCNN",
    "is_genome_valid",
    "validate_and_fix_genome",
    "calculate_max_safe_conv_layers",
    # Genetics
    "create_random_genome",
    "mutate_genome",
    "crossover_genomes",
    "select_population",
    "select_parents",
    "calculate_compatibility_distance",
    "assign_species",
    "assign_to_species",
    "INNOVATION_NAMESPACE",
    "innovation_uuid",
    "build_innovation_genes",
    "append_structural_event",
    # Evolution
    "HybridNeuroevolution",
    "evaluate_fitness",
    "train_fold_in_thread",
    "load_fold_data",
    # Evaluation
    "calculate_metrics",
    "aggregate_fold_metrics",
    "ArtifactManager",
    "evaluate_single_fold",
    "evaluate_5fold_cross_validation",
    # Visualization
    "plot_fitness_evolution",
    "show_evolution_statistics",
    "analyze_failed_evaluations",
    "configure_plot_style",
    "display_best_architecture",
    "print_checkpoint_info",
]
