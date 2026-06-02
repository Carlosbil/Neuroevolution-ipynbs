"""
Configuration management for neuroevolution experiments.

Provides default CONFIG dictionary and validation functions.
"""

import os
import torch.nn as nn
import torch.optim as optim


METRIC_ALIASES = {
    'recall': 'sensitivity',
}

SUPPORTED_METRIC_NAMES = {
    'accuracy',
    'precision',
    'sensitivity',
    'recall',
    'specificity',
    'f1_score',
    'auc',
}


def canonical_metric_name(metric_name: str) -> str:
    """Returns the canonical name for a supported classification metric."""
    normalized = str(metric_name).strip().lower()
    return METRIC_ALIASES.get(normalized, normalized)


def get_activation_functions() -> dict:
    """Returns mapping of activation function names to PyTorch classes."""
    return {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'selu': nn.SELU,
    }


def get_optimizers() -> dict:
    """Returns mapping of optimizer names to PyTorch classes."""
    return {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
    }


def get_default_config(info_path: str = None) -> dict:
    """
    Returns default configuration dictionary for neuroevolution.
    
    Args:
        info_path: Optional artifact directory path. Defaults to "artifacts/test_audio"
    
    Returns:
        Dictionary with all configuration parameters
    """
    if info_path is None:
        info_path = os.path.join("artifacts", "test_audio")
    
    return {
        # Genetic algorithm parameters
        'population_size': 20,
        'max_generations': 100,
        'fitness_threshold': 98.0,
        
        # Adaptive mutation parameters
        'base_mutation_rate': 0.25,
        'mutation_rate_min': 0.10,
        'mutation_rate_max': 0.80,
        'current_mutation_rate': 0.25,
        'structural_mutation_generation_factor': 0.5,
        
        'crossover_rate': 0.99,
        'elite_percentage': 0.2,

        # Selection strategy. Fitness-only mode preserves the historical scalar
        # behavior; Pareto mode ranks quality/cost tradeoffs for reproduction.
        'selection_strategy': 'pareto',
        'pareto_objectives': [
            {'name': 'fitness', 'direction': 'maximize'},
            {'name': 'evaluation_time_seconds', 'direction': 'minimize'},
            {'name': 'parameter_count', 'direction': 'minimize'},
        ],
        'pareto_tie_breaker': 'crowding_distance',
        'pareto_fitness_epsilon': 0.0,
        
        # Dataset selection
        'dataset': 'AUDIO',
        
        # Dataset parameters for audio
        'num_channels': 1,
        'sequence_length': 240000,
        'num_classes': 2,
        'batch_size': 64,
        'test_split': 0.2,
        
        # Training parameters
        'num_epochs': 100,
        'learning_rate': 0.00001,
        'early_stopping_patience': 100000,
        'use_amp': True,
        'amp_dtype': 'float16',
        'validation_frequency_epochs': 2,
        'fitness_metric': 'f1_score',
        'fold_selection_metric': 'fitness_metric',
        'metric_improvement_threshold': None,

        # Fold evaluation and data loading performance
        'fold_parallel_workers': 5,
        'fold_cache_mode': 'ram',  # Options: 'none', 'ram', 'memmap'
        'dataloader_num_workers': None,  # Auto when None
        'dataloader_persistent_workers': True,
        'dataloader_prefetch_factor': 2,
        'dataloader_pin_memory': True,
        
        # Epoch-level early stopping
        'epoch_patience': 10,
        'improvement_threshold': 0.01,
        
        # Generation-level early stopping
        'early_stopping_generations': 20,
        'min_improvement_threshold': 0.01,
        
        # Architecture range for 1D Conv
        'min_conv_layers': 1,
        'max_conv_layers': 30,
        'min_fc_layers': 1,
        'max_fc_layers': 10,
        'min_filters': 1,
        'max_filters': 256,
        'min_fc_nodes': 64,
        'max_fc_nodes': 1024,
        
        # Mutation parameters - Kernel sizes for 1D Conv
        'kernel_size_options': [1, 3, 5, 7, 9, 11, 13, 15],
        
        # Mutation parameters - Dropout range
        'min_dropout': 0.2,
        'max_dropout': 0.6,
        
        # Mutation parameters - Learning rate options
        'learning_rate_options': [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.01, 0.1, 0.00001],
        
        # Mutation parameters - Normalization type weights
        'normalization_batch_weight': 0.8,
        'normalization_layer_weight': 0.2,

        # Residual Conv1D search parameters. Residual mode is optional and
        # sampled as part of the architecture search space.
        'residual_enabled_weight': 0.35,
        'residual_disabled_weight': 0.65,
        'residual_block_size_options': [2, 3],
        'residual_projection_options': ['auto'],
        'residual_mutation_weight': 0.15,
        'max_model_parameters': None,

        # Inception Conv1D search parameters. The topology weights select one
        # active Conv1D topology per genome.
        'conv_topology_weights': {
            'sequential': 0.45,
            'residual': 0.30,
            'inception': 0.25,
        },
        'inception_reduction_ratio_options': [0.25, 0.5],
        'inception_pool_branch_options': [True, False],
        'inception_min_branch_channels': 1,
        'inception_mutation_weight': 0.15,

        # Known architecture templates. Templates are genome-level Conv1D
        # adaptations, not fixed imported reference models.
        'architecture_template_seed_fraction': 0.15,
        'architecture_template_mutation_weight': 0.05,
        'architecture_template_ids': [
            'lenet_conv1d_tiny',
            'alexnet_conv1d_small',
            'vgg_conv1d_small',
            'resnet_conv1d_small',
            'resnet_conv1d_medium',
            'wide_resnet_conv1d_small',
            'googlenet_inception_conv1d_small',
            'googlenet_inception_conv1d_medium',
            'inception_conv1d_wide',
        ],
        'architecture_template_seed_min_random_fraction': 0.50,
        'architecture_template_max_attempts': 20,
        
        'artifact_dir': info_path,
        'artifacts_dir': info_path,
        
        # Audio dataset configuration (OS-independent paths)
        'dataset_id': '40_1e5_N',
        'fold_id': '40_1e5_N',
        'num_folds': 5,
        'data_path': os.path.join('data', 'sets', 'folds_5'),
        'fold_files_subdirectory': 'files_real_40_1e5_N',
        'normalization': {'mean': (0.0,), 'std': (1.0,)}
    }


def validate_config(config: dict) -> None:
    """
    Validates configuration dictionary for consistency.
    
    Args:
        config: Configuration dictionary to validate
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Population and generation constraints
    if config['population_size'] < 2:
        raise ValueError("population_size must be at least 2")
    if config['max_generations'] < 1:
        raise ValueError("max_generations must be at least 1")
    
    # Mutation rate bounds
    if not (0 <= config['mutation_rate_min'] <= config['mutation_rate_max'] <= 1):
        raise ValueError("Mutation rates must satisfy: 0 <= min <= max <= 1")
    if not (config['mutation_rate_min'] <= config['base_mutation_rate'] <= config['mutation_rate_max']):
        raise ValueError("base_mutation_rate must be between min and max")
    
    # Elite percentage
    if not (0 <= config['elite_percentage'] <= 1):
        raise ValueError("elite_percentage must be between 0 and 1")

    fitness_metric = str(config.get('fitness_metric', 'f1_score')).strip().lower()
    if fitness_metric not in SUPPORTED_METRIC_NAMES:
        valid_options = ', '.join(sorted(SUPPORTED_METRIC_NAMES))
        raise ValueError(f"fitness_metric must be one of: {valid_options}")

    fold_selection_metric = str(config.get('fold_selection_metric', 'fitness_metric')).strip().lower()
    if fold_selection_metric != 'fitness_metric' and fold_selection_metric not in SUPPORTED_METRIC_NAMES:
        valid_options = ', '.join(sorted(SUPPORTED_METRIC_NAMES | {'fitness_metric'}))
        raise ValueError(f"fold_selection_metric must be one of: {valid_options}")

    metric_improvement_threshold = config.get('metric_improvement_threshold')
    if metric_improvement_threshold is not None and float(metric_improvement_threshold) < 0:
        raise ValueError("metric_improvement_threshold must be non-negative or None")

    # Selection strategy
    selection_strategy = str(config.get('selection_strategy', 'pareto')).lower()
    if selection_strategy not in {'fitness', 'pareto'}:
        raise ValueError("selection_strategy must be one of: 'fitness', 'pareto'")

    supported_objectives = {'fitness', 'evaluation_time_seconds', 'parameter_count'}
    supported_directions = {'maximize', 'minimize'}
    pareto_objectives = config.get('pareto_objectives', [])
    if not isinstance(pareto_objectives, list) or not pareto_objectives:
        raise ValueError("pareto_objectives must be a non-empty list")

    seen_objectives = set()
    for objective in pareto_objectives:
        if not isinstance(objective, dict):
            raise ValueError("Each Pareto objective must be a dictionary")
        objective_name = objective.get('name')
        direction = objective.get('direction')
        if objective_name not in supported_objectives:
            raise ValueError(f"Unsupported Pareto objective: {objective_name}")
        if direction not in supported_directions:
            raise ValueError("Pareto objective direction must be 'maximize' or 'minimize'")
        if objective_name in seen_objectives:
            raise ValueError(f"Duplicate Pareto objective: {objective_name}")
        seen_objectives.add(objective_name)

    if float(config.get('pareto_fitness_epsilon', 0.0)) < 0:
        raise ValueError("pareto_fitness_epsilon must be non-negative")
    
    # Architecture constraints
    if config['min_conv_layers'] < 1 or config['max_conv_layers'] < config['min_conv_layers']:
        raise ValueError("Invalid conv layer bounds")
    if config['min_fc_layers'] < 1 or config['max_fc_layers'] < config['min_fc_layers']:
        raise ValueError("Invalid FC layer bounds")
    if config['min_filters'] < 1 or config['max_filters'] < config['min_filters']:
        raise ValueError("Invalid filter bounds")
    if config['min_fc_nodes'] < 1 or config['max_fc_nodes'] < config['min_fc_nodes']:
        raise ValueError("Invalid FC node bounds")
    
    # Dataset parameters
    if config['num_channels'] < 1:
        raise ValueError("num_channels must be at least 1")
    if config['num_classes'] < 2:
        raise ValueError("num_classes must be at least 2")
    if config['batch_size'] < 1:
        raise ValueError("batch_size must be at least 1")

    # Performance-related parameters
    if int(config.get('validation_frequency_epochs', 1)) < 1:
        raise ValueError("validation_frequency_epochs must be at least 1")
    if int(config.get('fold_parallel_workers', 1)) < 1:
        raise ValueError("fold_parallel_workers must be at least 1")

    fold_cache_mode = str(config.get('fold_cache_mode', 'ram')).lower()
    if fold_cache_mode not in {'none', 'ram', 'memmap'}:
        raise ValueError("fold_cache_mode must be one of: 'none', 'ram', 'memmap'")

    dataloader_num_workers = config.get('dataloader_num_workers')
    if dataloader_num_workers is not None and int(dataloader_num_workers) < 0:
        raise ValueError("dataloader_num_workers must be >= 0 or None")

    if int(config.get('dataloader_prefetch_factor', 1)) < 1:
        raise ValueError("dataloader_prefetch_factor must be at least 1")

    # Residual search parameters
    residual_enabled_weight = float(config.get('residual_enabled_weight', 0.0))
    residual_disabled_weight = float(config.get('residual_disabled_weight', 1.0))
    if residual_enabled_weight < 0 or residual_disabled_weight < 0:
        raise ValueError("Residual topology weights must be non-negative")
    if residual_enabled_weight + residual_disabled_weight <= 0:
        raise ValueError("At least one residual topology weight must be positive")

    residual_block_sizes = config.get('residual_block_size_options', [])
    if not residual_block_sizes or any(int(size) < 1 for size in residual_block_sizes):
        raise ValueError("residual_block_size_options must contain positive integers")

    residual_projection_options = config.get('residual_projection_options', [])
    if not residual_projection_options or 'auto' not in residual_projection_options:
        raise ValueError("residual_projection_options must include 'auto'")

    residual_mutation_weight = float(config.get('residual_mutation_weight', 0.0))
    if not (0 <= residual_mutation_weight <= 1):
        raise ValueError("residual_mutation_weight must be between 0 and 1")

    max_model_parameters = config.get('max_model_parameters')
    if max_model_parameters is not None and int(max_model_parameters) < 1:
        raise ValueError("max_model_parameters must be a positive integer or None")

    # Conv topology and Inception search parameters
    topology_weights = config.get('conv_topology_weights')
    if topology_weights is not None:
        supported_topologies = {'sequential', 'residual', 'inception'}
        unknown_topologies = set(topology_weights) - supported_topologies
        if unknown_topologies:
            raise ValueError("conv_topology_weights contains unsupported topology names")
        topology_weight_sum = 0.0
        for topology in supported_topologies:
            weight = float(topology_weights.get(topology, 0.0))
            if weight < 0:
                raise ValueError("conv_topology_weights values must be non-negative")
            topology_weight_sum += weight
        if topology_weight_sum <= 0:
            raise ValueError("At least one conv_topology_weights value must be positive")

    reduction_options = config.get('inception_reduction_ratio_options', [])
    if not reduction_options or any(float(ratio) <= 0 or float(ratio) > 1 for ratio in reduction_options):
        raise ValueError("inception_reduction_ratio_options must contain values in (0, 1]")

    pool_options = config.get('inception_pool_branch_options', [])
    if not pool_options or any(not isinstance(option, bool) for option in pool_options):
        raise ValueError("inception_pool_branch_options must contain booleans")

    if int(config.get('inception_min_branch_channels', 1)) < 1:
        raise ValueError("inception_min_branch_channels must be at least 1")

    inception_mutation_weight = float(config.get('inception_mutation_weight', 0.0))
    if not (0 <= inception_mutation_weight <= 1):
        raise ValueError("inception_mutation_weight must be between 0 and 1")

    # Known architecture template parameters
    template_seed_fraction = float(config.get('architecture_template_seed_fraction', 0.0))
    if not (0 <= template_seed_fraction <= 1):
        raise ValueError("architecture_template_seed_fraction must be between 0 and 1")

    template_mutation_weight = float(config.get('architecture_template_mutation_weight', 0.0))
    if not (0 <= template_mutation_weight <= 1):
        raise ValueError("architecture_template_mutation_weight must be between 0 and 1")

    min_random_fraction = float(config.get('architecture_template_seed_min_random_fraction', 0.0))
    if not (0 <= min_random_fraction <= 1):
        raise ValueError("architecture_template_seed_min_random_fraction must be between 0 and 1")

    template_max_attempts = int(config.get('architecture_template_max_attempts', 1))
    if template_max_attempts < 1:
        raise ValueError("architecture_template_max_attempts must be at least 1")

    template_ids = config.get('architecture_template_ids', [])
    if not isinstance(template_ids, list) or not template_ids:
        raise ValueError("architecture_template_ids must be a non-empty list")
    if len(template_ids) != len(set(template_ids)):
        raise ValueError("architecture_template_ids must not contain duplicate template IDs")

    from neuroevolution.genetics.architecture_templates import get_template_registry

    known_template_ids = set(get_template_registry())
    unknown_template_ids = set(template_ids) - known_template_ids
    if unknown_template_ids:
        unknown = ', '.join(sorted(unknown_template_ids))
        raise ValueError(f"unknown architecture template IDs: {unknown}")


# Global constants - exported for convenience
ACTIVATION_FUNCTIONS = get_activation_functions()
OPTIMIZERS = get_optimizers()
REQUIRED_PACKAGES = [
    "torch==2.11.0",
    "torchvision==0.26.0",
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "tqdm>=4.64.0",
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0",
    "scikit-learn"
]

# Default config constant for convenience imports
CONFIG = get_default_config()
