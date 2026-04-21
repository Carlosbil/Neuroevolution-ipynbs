"""
Configuration management for neuroevolution experiments.

Provides default CONFIG dictionary and validation functions.
"""

import os
import torch.nn as nn
import torch.optim as optim


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
        
        'crossover_rate': 0.99,
        'elite_percentage': 0.2,
        
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
