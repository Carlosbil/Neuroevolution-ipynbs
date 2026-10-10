"""
Configuration management for neuroevolution experiments.

Provides default CONFIG dictionary and validation functions.
"""

import os
import copy
from pathlib import Path
import numpy as np
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

VOWELS_INPUTS = {
    'audio': {'array_directory': 'arrays_audio', 'batch_size': 2, 'pre_fc_pool_length': 64},
    'spectrogram': {'array_directory': 'arrays', 'batch_size': 8, 'pre_fc_pool_shape': (4, 8)},
}

# All experiment choices live here. Change this value to switch input profiles.
DEFAULT_INPUT_MODALITY = 'spectrogram'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ARCHITECTURE_IDS = (
    'lenet5', 'resnet18', 'mobilenet_v2', 'efficientnet_b0', 'alexnet',
    'googlenet', 'densenet121', 'vgg16',
    'vgg11', 'vgg13', 'vgg19',
    'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
    'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'wide_resnet50_2', 'wide_resnet101_2',
    'resnext50_32x4d', 'resnext101_32x8d',
    'inception_v3',
    'densenet161', 'densenet169', 'densenet201',
    'mobilenet_v3_large',
    'efficientnet_b1', 'efficientnet_b2',
    'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7',
    'squeezenet1_0', 'squeezenet1_1',
)


VOWELS_DATASET_CONFIG = {
    'seed': 42,
    'train_fraction': 0.70, 'validation_fraction': 0.10, 'test_fraction': 0.20,
    'num_folds': 5, 'duration_quantile': 0.95,
    'source_root': os.fspath(PROJECT_ROOT / 'new_data' / 'vowels'),
    'output_root': os.fspath(PROJECT_ROOT / 'artifacts' / 'vowels_dataset'),
    'n_mels': 64, 'n_fft': 1024, 'hop_length': 441, 'feature_id': 'vowels',
    'min_hz': 80.0, 'max_hz': 8000.0, 'log_mel_power_scale': 1000.0,
    'spectrogram_array_directory': 'arrays', 'audio_array_directory': 'arrays_audio',
}


# Keep the original notebook's output separate from the cropped experiment.
VOWELS_HALF_DATASET_CONFIG = {
    **VOWELS_DATASET_CONFIG,
    'output_root': os.fspath(PROJECT_ROOT / 'artifacts' / 'vowels_dataset_half'),
    'sample_rate_hz': 44_100,
    'normalized_duration_s': 6.5,
    'trim_left_s': 1.5,
    'trim_right_s': 1.5,
    'duration_s': 3.5,
    'audio_samples': 154_350,
    'spectrogram_frames': 348,
    'preparation_notebook': 'vowels_dataset_por_persona_half.ipynb',
}


def configure_vowels_input(config: dict, dataset_root: str = None) -> dict:
    """Validate configured arrays without changing any experiment option."""
    modality = str(config.get('input_modality', 'audio')).strip().lower()
    # Preserve notebooks configured with the earlier name.
    if modality == 'image':
        modality = 'spectrogram'
    if modality not in VOWELS_INPUTS:
        raise ValueError(f"input_modality must be one of: {', '.join(VOWELS_INPUTS)}")
    dataset_root = os.fspath(dataset_root or config['data_path'])
    if Path(dataset_root).resolve() != Path(config['data_path']).resolve():
        raise ValueError('Set data_path in config.py; the notebook cannot replace it')
    array_directory = config['fold_files_subdirectory']
    first_fold = os.path.join(dataset_root, array_directory, 'X_train_vowels_fold_1.npy')
    if not os.path.isfile(first_fold):
        notebook = config.get('dataset_preparation_notebook', VOWELS_HALF_DATASET_CONFIG['preparation_notebook'])
        raise FileNotFoundError(f"Run {notebook} first: {first_fold}")
    shape = np.load(first_fold, mmap_mode='r').shape
    if len(shape) != 3 or shape[0] < 1:
        raise ValueError(f"Unexpected vowels fold shape: {shape}")
    expected_tail = (config['num_channels'], config['sequence_length']) if modality == 'audio' else (config['num_frequency_bins'], config['sequence_length'])
    if tuple(shape[1:]) != expected_tail:
        raise ValueError(f"Unexpected {modality} input shape: {shape}; expected (samples, *{expected_tail})")

    return config


def resolve_config(config: dict) -> dict:
    """Copy canonical defaults and explicit options; never mutate the caller."""
    resolved = get_default_config()
    resolved.update(copy.deepcopy(config))
    return resolved


def get_operator_config(config: dict, generation: int, mutation_rate: float,
                        conv_cap: int = None, fc_cap: int = None) -> dict:
    """Derive transient operator inputs separately from experiment configuration."""
    stage = max(0, generation // max(1, int(config['complexity_step_generations'])))
    runtime = dict(config)
    runtime.update({
        'current_generation': generation,
        'current_mutation_rate': mutation_rate,
        'current_max_conv_layers': min(config['max_conv_layers'], config['initial_max_conv_layers'] + stage) if conv_cap is None else conv_cap,
        'current_max_fc_layers': min(config['max_fc_layers'], config['initial_max_fc_layers'] + stage) if fc_cap is None else fc_cap,
    })
    return runtime


def with_fold_progress_callback(config: dict, callback) -> dict:
    """Attach scheduling state to a copy, without overwriting experiment options."""
    return {**config, '_fold_progress_callback': callback}


def with_fold_epoch_callback(config: dict, callback) -> dict:
    """Attach an observer without changing experiment settings."""
    return {**config, '_fold_epoch_callback': callback}


def optimizer_learning_rates(config: dict, optimizer: str) -> list:
    return list(config.get('optimizer_learning_rate_options', {}).get(optimizer, config['learning_rate_options']))


def config_value(config: dict, key: str):
    """Read an explicit setting or its canonical default from this module."""
    return config[key] if key in config else CONFIG[key]


def configured_epochs(config: dict, requested=None) -> int:
    """Accept legacy arguments only when they agree with config.py."""
    epochs = int(config_value(config, 'num_epochs'))
    if requested is not None and requested != epochs:
        raise ValueError('Set num_epochs in config.py; function arguments cannot override it')
    return epochs


def reference_input_size(config: dict, architecture_id: str) -> tuple:
    if architecture_id == 'lenet5':
        return tuple(config['reference_lenet_input_size'])
    if architecture_id == 'inception_v3':
        return tuple(config['reference_inception_input_size'])
    return tuple(config.get('reference_input_sizes', {}).get(architecture_id, config['reference_input_size']))


def make_optimizer(model, genome: dict, config: dict):
    """Use the same optimizer settings in search and final evaluation."""
    name = genome['optimizer']
    options = copy.deepcopy(config.get('optimizer_kwargs', {}).get(name, {}))
    return OPTIMIZERS[name](model.parameters(), lr=genome['learning_rate'], **options)


def canonical_metric_name(metric_name: str) -> str:
    """Returns the canonical name for a supported classification metric."""
    normalized = str(metric_name).strip().lower()
    return METRIC_ALIASES.get(normalized, normalized)


def get_data_phase_config(config: dict, phase: str) -> dict:
    """Return a copy of *config* bound to the data source for one protocol phase.

    Each phase uses the source selectors declared in config.py, without
    mutating the caller's configuration between the two phases.
    """
    normalized_phase = str(phase).strip().lower()
    selectors = {
        'evolution': ('evolution_dataset_id', 'evolution_fold_files_subdirectory'),
        'final_evaluation': ('final_dataset_id', 'final_fold_files_subdirectory'),
    }
    if normalized_phase not in selectors:
        valid_phases = ', '.join(sorted(selectors))
        raise ValueError(f"phase must be one of: {valid_phases}")

    dataset_key, directory_key = selectors[normalized_phase]
    phase_config = dict(config)
    phase_config['dataset_id'] = config.get(dataset_key, config.get('dataset_id'))
    phase_config['fold_files_subdirectory'] = config.get(
        directory_key,
        config.get('fold_files_subdirectory'),
    )
    return phase_config


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
        info_path: Optional artifact directory path for the configured modality.
    
    Returns:
        Dictionary with all configuration parameters
    """
    if info_path is None:
        info_path = os.fspath(PROJECT_ROOT / 'artifacts' / f'test_vowels_half_{DEFAULT_INPUT_MODALITY}')
    
    modality = DEFAULT_INPUT_MODALITY
    profile = VOWELS_INPUTS[modality]
    dataset_config = VOWELS_HALF_DATASET_CONFIG
    return {
        'seed': 42,
        'resume_evolution': False,
        'speciation_enabled': True,
        'species_elite_fraction': 0.15,
        'random_immigrant_fraction': 0.10,
        'reference_injection_fraction': 0.10,
        'adaptive_mutation_strength': 0.4,
        'mutation_stagnation_boost': 0.15,
        'conv_dropout_rate': 0.1,
        'activation_options': ['relu', 'leaky_relu', 'tanh'],
        'tournament_size': 3,
        'minimum_specificity': 1.0,
        'minimum_sensitivity': 1.0,
        'mutation_max_attempts': 50,
        'crossover_max_attempts': 20,
        'random_genome_max_attempts': 100,
        'minimum_feature_map_size': 4,
        'reference_max_model_parameters': 150_000_000,
        'reference_seed_fallback_to_random': False,
        'reference_input_size': (224, 224),
        # EfficientNet compound scaling includes its standard input resolution.
        'reference_input_sizes': {
            'efficientnet_b0': (224, 224), 'efficientnet_b1': (240, 240),
            'efficientnet_b2': (260, 260), 'efficientnet_b3': (300, 300),
            'efficientnet_b4': (380, 380), 'efficientnet_b5': (456, 456),
            'efficientnet_b6': (528, 528), 'efficientnet_b7': (600, 600),
        },
        'reference_lenet_input_size': (32, 32),
        'reference_inception_input_size': (299, 299),
        'reference_aux_logits': True,
        'reference_aux_loss_weight': 0.3,
        'require_all_folds': True,
        'reference_dropout_rate': None,
        'reference_seed_optimizer': 'adamw',
        'reference_seed_learning_rate': 1e-4,
        'template_seed_optimizer': 'adam',
        'template_seed_learning_rate': 1e-3,
        'optimizer_options': ['adamw', 'adam', 'rmsprop', 'sgd'],
        'optimizer_learning_rate_options': {
            'adam': [1e-3, 5e-4, 2e-4, 1e-4, 5e-5],
            'adamw': [1e-3, 5e-4, 2e-4, 1e-4, 5e-5],
            'rmsprop': [1e-3, 5e-4, 2e-4, 1e-4],
            'sgd': [0.1, 0.03, 0.01, 0.003, 0.001],
        },
        'optimizer_kwargs': {
            'adam': {'weight_decay': 1e-4},
            'adamw': {'weight_decay': 1e-2},
            'rmsprop': {'weight_decay': 1e-4, 'momentum': 0.9},
            'sgd': {'weight_decay': 1e-4, 'momentum': 0.9, 'nesterov': True},
        },
        'max_batches_per_epoch': None,
        # Genetic algorithm parameters
        # Tuned for the available H100 NVL MIG 3g.47gb CUDA partition.
        # Every individual contributes five folds to one global work queue.
        # Search-quality preset: more independent candidates and a long enough
        # horizon for speciation and incremental growth to produce useful variants.
        'population_size': 20,
        'max_generations': 100,
        'fitness_threshold': 100.0,
        
        # Adaptive mutation parameters
        'base_mutation_rate': 0.20,
        'mutation_rate_min': 0.05,
        'mutation_rate_max': 0.50,
        'current_mutation_rate': 0.20,
        # Prevent late generations from making destructive, very large depth jumps.
        'structural_mutation_generation_factor': 0.03,
        
        'crossover_rate': 0.85,
        'elite_percentage': 0.10,

        # Selection strategy. Fitness-only mode preserves the historical scalar
        # behavior; Pareto mode ranks quality/cost tradeoffs for reproduction.
        'selection_strategy': 'fitness',
        'pareto_objectives': [
            {'name': 'fitness', 'direction': 'maximize'},
            {'name': 'evaluation_time_seconds', 'direction': 'minimize'},
            {'name': 'parameter_count', 'direction': 'minimize'},
        ],
        'pareto_tie_breaker': 'crowding_distance',
        'pareto_fitness_epsilon': 0.0,
        
        # Dataset selection
        'dataset': 'AUDIO',
        # Choose DEFAULT_INPUT_MODALITY above. Profiles are resolved here only.
        'input_modality': modality,
        'pre_fc_pool_length': profile.get('pre_fc_pool_length'),
        'pre_fc_pool_shape': profile.get('pre_fc_pool_shape'),
        'num_frequency_bins': dataset_config['n_mels'] if modality == 'spectrogram' else None,
        
        # Dataset parameters for audio
        'num_channels': 1,
        # Explicit input dimensions; loaders validate them and never rewrite them.
        'sequence_length': dataset_config['spectrogram_frames'] if modality == 'spectrogram' else dataset_config['audio_samples'],
        'dataset_preparation_notebook': dataset_config['preparation_notebook'],
        'audio_duration_s': dataset_config['duration_s'],
        'audio_sample_rate_hz': dataset_config['sample_rate_hz'],
        'audio_trim_left_s': dataset_config['trim_left_s'],
        'audio_trim_right_s': dataset_config['trim_right_s'],
        'num_classes': 2,
        # Per-modality batch comes from the profile at the top of this file.
        'batch_size': profile['batch_size'],
        'test_split': 0.2,
        
        # Training parameters
        'num_epochs': 100,
        # 'genome' uses the evolved activations; 'relu' replays older checkpoints.
        'fc_activation_source': 'genome',
        'learning_rate': 0.0001,
        'use_amp': True,
        # H100: BF16 uses Tensor Cores with FP32-like exponent range, avoiding
        # the loss-scaling overhead required by FP16.
        'amp_dtype': 'bfloat16',
        'validation_frequency_epochs': 1,
        'fitness_metric': 'f1_score',
        'checkpoint_metric': 'f1_score',
        'fold_selection_metric': 'fitness_metric',
        'metric_improvement_threshold': None,

        # Number of folds trained concurrently across the entire population.
        # Use 1 for sequential execution, 5 for five simultaneous folds, or
        # another positive value if GPU memory permits it.
        'fold_parallel_workers': 3,
        'fold_cache_mode': 'ram' if modality == 'spectrogram' else 'none',  # Options: 'none', 'ram', 'memmap'
        # Fold arrays are cached in RAM as tensors. Extra worker processes add
        # IPC overhead (and multiply across the five parallel folds) without
        # performing decoding or augmentation work.
        'dataloader_num_workers': 0,
        'dataloader_persistent_workers': False,
        'dataloader_prefetch_factor': 2,
        'dataloader_pin_memory': True,
        
        # Epoch-level early stopping
        'epoch_patience': 25,
        'improvement_threshold': 0.01,
        
        # Generation-level early stopping
        'early_stopping_generations': 30,
        'min_improvement_threshold': 0.01,
        
        # Architecture range for 1D Conv
        'min_conv_layers': 1,
        'max_conv_layers': 11,
        'min_fc_layers': 1,
        'max_fc_layers': 4,
        'min_filters': 8,
        'max_filters': 256,
        'min_fc_nodes': 64,
        'max_fc_nodes': 768,

        # Search shallow models first, then unlock depth progressively. This keeps
        # early evaluations inexpensive while allowing the full safe search space.
        'initial_max_conv_layers': 3,
        'initial_max_fc_layers': 1,
        'complexity_step_generations': 8,
        'incremental_growth_probability': 0.7,
        'speciation_threshold': 0.45,
        'species_elite_min': 1,
        'species_survival_rate': 0.5,
        
        # Mutation parameters - Kernel sizes for 1D Conv
        'kernel_size_options': [3, 5, 7, 9, 11, 13, 15],
        
        # Mutation parameters - Dropout range
        'min_dropout': 0.1,
        'max_dropout': 0.5,
        
        # Mutation parameters - Learning rate options
        'learning_rate_options': [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001],
        
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
        'max_model_parameters': 20_000_000,

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

        # Complete reference networks are independent of random-genome caps.
        # Legacy reduced Conv1D IDs remain loadable, but are not default seeds.
        'architecture_template_seed_fraction': 0.40,
        'architecture_template_mutation_weight': 0.05,
        'architecture_template_ids': list(REFERENCE_ARCHITECTURE_IDS) if modality == 'spectrogram' else [],
        'architecture_template_seed_min_random_fraction': 0.50,
        'architecture_template_max_attempts': 20,
        
        'artifact_dir': info_path,
        'artifacts_dir': info_path,
        
        # Vowels: internal folds use train subjects; final global held-out test
        # is evaluated only after architecture selection in test.ipynb.
        'evolution_dataset_id': dataset_config['feature_id'],
        'evolution_fold_files_subdirectory': profile['array_directory'],
        'final_dataset_id': dataset_config['feature_id'],
        'final_fold_files_subdirectory': profile['array_directory'],
        'dataset_id': dataset_config['feature_id'],
        'fold_id': 'N',
        'num_folds': dataset_config['num_folds'],
        'data_path': dataset_config['output_root'],
        'fold_files_subdirectory': profile['array_directory'],
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
    for key in ('species_elite_fraction', 'random_immigrant_fraction', 'reference_injection_fraction', 'species_survival_rate'):
        if not 0 <= float(config[key]) <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if config.get('fc_activation_source', 'genome') not in {'genome', *ACTIVATION_FUNCTIONS}:
        raise ValueError('fc_activation_source must be genome or a supported activation')
    if config['tournament_size'] < 1 or config['epoch_patience'] < 1 or config['early_stopping_generations'] < 1:
        raise ValueError('Tournament size and patience must be positive')
    for key in ('minimum_specificity', 'minimum_sensitivity'):
        if not 0 <= float(config[key]) <= 100:
            raise ValueError(f"{key} must be a percentage in [0, 100]")
    for name in config['optimizer_options']:
        if name not in get_optimizers():
            raise ValueError(f"Unsupported optimizer: {name}")
        if not optimizer_learning_rates(config, name) or any(rate <= 0 for rate in optimizer_learning_rates(config, name)):
            raise ValueError(f"Learning rates for {name} must be positive")
    if not config['optimizer_options']:
        raise ValueError('optimizer_options must not be empty')
    for prefix in ('reference', 'template'):
        name, rate = config[f'{prefix}_seed_optimizer'], config[f'{prefix}_seed_learning_rate']
        if name not in config['optimizer_options'] or rate not in optimizer_learning_rates(config, name):
            raise ValueError(f'{prefix} seed optimizer and learning rate must be configured search options')
    if any(int(config[key]) < 1 for key in ('random_genome_max_attempts', 'mutation_max_attempts', 'crossover_max_attempts', 'minimum_feature_map_size')):
        raise ValueError('Operator attempt limits and minimum_feature_map_size must be positive')
    if config['max_batches_per_epoch'] is not None and int(config['max_batches_per_epoch']) < 1:
        raise ValueError('max_batches_per_epoch must be positive or None')
    if config['reference_max_model_parameters'] is not None and config['reference_max_model_parameters'] < 1:
        raise ValueError('reference_max_model_parameters must be positive or None')
    if any(name not in ACTIVATION_FUNCTIONS for name in config['activation_options']) or not config['activation_options']:
        raise ValueError('activation_options must contain supported activations')
    if not 0 <= config['conv_dropout_rate'] < 1:
        raise ValueError('conv_dropout_rate must be in [0, 1)')
    if config['reference_aux_loss_weight'] < 0:
        raise ValueError('reference_aux_loss_weight must be nonnegative')
    if config['reference_dropout_rate'] is not None and not 0 <= config['reference_dropout_rate'] < 1:
        raise ValueError('reference_dropout_rate must be in [0, 1) or None')

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
    modality = config.get('input_modality')
    if modality is not None and modality not in {*VOWELS_INPUTS, 'image'}:
        raise ValueError(f"input_modality must be one of: {', '.join(VOWELS_INPUTS)}")
    pool_length = config.get('pre_fc_pool_length')
    if pool_length is not None and int(pool_length) < 1:
        raise ValueError('pre_fc_pool_length must be positive or None')
    pool_shape = config.get('pre_fc_pool_shape')
    if pool_shape is not None and (len(pool_shape) != 2 or any(int(axis) < 1 for axis in pool_shape)):
        raise ValueError('pre_fc_pool_shape must contain two positive dimensions')
    if modality in {'spectrogram', 'image'} and int(config.get('num_frequency_bins') or 0) < 4:
        raise ValueError('Spectrogram mode needs at least four frequency bins')
    if config['num_channels'] < 1:
        raise ValueError("num_channels must be at least 1")
    if config['num_classes'] < 2:
        raise ValueError("num_classes must be at least 2")
    if config['batch_size'] < 1:
        raise ValueError("batch_size must be at least 1")

    for key in (
        'evolution_dataset_id',
        'evolution_fold_files_subdirectory',
        'final_dataset_id',
        'final_fold_files_subdirectory',
    ):
        if key in config and not str(config[key]).strip():
            raise ValueError(f"{key} must be a non-empty string")

    # Performance-related parameters
    if int(config.get('validation_frequency_epochs', 1)) < 1:
        raise ValueError("validation_frequency_epochs must be at least 1")
    if str(config.get('amp_dtype', 'float16')).lower() not in {'float16', 'bfloat16'}:
        raise ValueError("amp_dtype must be either 'float16' or 'bfloat16'")
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
    if not isinstance(template_ids, list):
        raise ValueError('architecture_template_ids must be a list')
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
    # Keep the native libraries on the same supported release pair.
    "torch==2.11.0",
    "torchvision==0.26.0",
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "tqdm>=4.64.0",
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0",
    "scikit-learn",
    "packaging>=23.0",
]

# Default config constant for convenience imports
CONFIG = get_default_config()


def get_winner_replay_config() -> dict:
    """One sequential CV pass of the uploaded winner using its genome activations."""
    config = get_default_config()
    output = os.fspath(PROJECT_ROOT / 'artifacts' / 'winner_68e15b97_genome_activations')
    config.update({
        'winner_progress_path': 'C:/Users/carlo/Downloads/evolution_progress.json',
        'winner_expected_id': '68e15b97',
        'fold_parallel_workers': 1,
        'num_epochs': 100,
        'epoch_patience': 18,
        'fc_activation_source': 'genome',
        'seed': 42,
        'fold_seed_stride': 1,
        'torch_num_threads': 8,
        'device': 'cuda',
        'artifact_dir': output,
        'artifacts_dir': output,
        'fold_cache_mode': 'none',
    })
    return config


def get_scientific_evaluation_config() -> dict:
    """Analysis of saved models only; no additional training or tuning."""
    replay = get_winner_replay_config()
    return {
        'run_results_path': os.fspath(Path(replay['artifacts_dir']) / 'results.json'),
        'report_directory': os.fspath(Path(replay['artifacts_dir']) / 'scientific_report'),
        'bootstrap_repetitions': 2000,
        'confidence_level': 0.95,
        'bootstrap_seed': 42,
        'person_probability_threshold': 0.5,
        'person_aggregation': 'mean_probability',
        'test_model_selection_metric': 'f1_score',
        'figure_dpi': 180,
        'figure_format': 'png',
    }
