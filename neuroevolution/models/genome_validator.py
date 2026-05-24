"""
Genome validation utilities for architecture safety checks.
"""

import math
import numpy as np
import random


DEFAULT_RESIDUAL_BLOCK_SIZE = 2
DEFAULT_RESIDUAL_PROJECTION = "auto"


def _get_residual_block_options(config: dict) -> list:
    """Returns supported residual block sizes as positive integers."""
    options = config.get('residual_block_size_options', [DEFAULT_RESIDUAL_BLOCK_SIZE])
    normalized = sorted({int(option) for option in options if int(option) > 0})
    return normalized or [DEFAULT_RESIDUAL_BLOCK_SIZE]


def _get_residual_projection_options(config: dict) -> list:
    """Returns supported residual projection modes."""
    options = config.get('residual_projection_options', [DEFAULT_RESIDUAL_PROJECTION])
    normalized = [str(option) for option in options]
    return normalized or [DEFAULT_RESIDUAL_PROJECTION]


def _as_bool(value) -> bool:
    """Converts common persisted boolean values without treating 'False' as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def normalize_residual_fields(genome: dict, config: dict) -> dict:
    """
    Adds and normalizes residual topology fields in-place.

    Residual mode is a searchable topology choice, not a requirement. It is
    only active when enough convolutional layers exist to form a residual
    block.
    """
    block_options = _get_residual_block_options(config)
    projection_options = _get_residual_projection_options(config)

    residual_enabled = _as_bool(genome.get('residual_enabled', False))
    residual_block_size = int(genome.get('residual_block_size', DEFAULT_RESIDUAL_BLOCK_SIZE))
    residual_projection = str(genome.get('residual_projection', DEFAULT_RESIDUAL_PROJECTION))

    if residual_block_size not in block_options:
        residual_block_size = block_options[0]

    num_conv_layers = int(genome.get('num_conv_layers', 0))
    if residual_enabled:
        valid_for_depth = [size for size in block_options if size <= num_conv_layers]
        if valid_for_depth:
            if residual_block_size > num_conv_layers:
                residual_block_size = valid_for_depth[-1]
        else:
            residual_enabled = False

    if residual_projection not in projection_options:
        residual_projection = projection_options[0]

    genome['residual_enabled'] = residual_enabled
    genome['residual_block_size'] = residual_block_size
    genome['residual_projection'] = residual_projection
    return genome


def calculate_pooling_operation_count(
    num_conv_layers: int,
    residual_enabled: bool = False,
    residual_block_size: int = DEFAULT_RESIDUAL_BLOCK_SIZE,
) -> int:
    """Returns how many MaxPool1d operations a conv stack applies."""
    num_conv_layers = max(0, int(num_conv_layers))
    residual_block_size = max(1, int(residual_block_size))
    if residual_enabled:
        return int(math.ceil(num_conv_layers / residual_block_size))
    return num_conv_layers


def estimate_genome_parameter_count(genome: dict, config: dict) -> int:
    """
    Deterministically estimates trainable parameters without instantiating a model.
    """
    fixed = normalize_residual_fields(dict(genome), config)
    num_conv_layers = int(fixed['num_conv_layers'])
    filters = list(fixed.get('filters', []))[:num_conv_layers]
    kernel_sizes = list(fixed.get('kernel_sizes', []))[:num_conv_layers]
    fc_nodes = list(fixed.get('fc_nodes', []))[:int(fixed.get('num_fc_layers', 0))]

    in_channels = int(config['num_channels'])
    conv_params = 0
    block_start_channels = in_channels
    residual_enabled = bool(fixed.get('residual_enabled', False))
    residual_block_size = int(fixed.get('residual_block_size', DEFAULT_RESIDUAL_BLOCK_SIZE))

    for index, out_channels in enumerate(filters):
        kernel_size = int(kernel_sizes[index]) if index < len(kernel_sizes) else 3
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        out_channels = int(out_channels)
        conv_params += (in_channels * out_channels * kernel_size) + out_channels
        conv_params += 2 * out_channels
        in_channels = out_channels

        is_block_end = (
            residual_enabled
            and ((index + 1) % residual_block_size == 0 or index == num_conv_layers - 1)
        )
        if is_block_end:
            if block_start_channels != out_channels:
                conv_params += (block_start_channels * out_channels) + out_channels
            block_start_channels = out_channels

    pool_count = calculate_pooling_operation_count(
        num_conv_layers,
        residual_enabled=residual_enabled,
        residual_block_size=residual_block_size,
    )
    temporal_length = int(config['sequence_length'])
    for _ in range(pool_count):
        temporal_length = max(1, temporal_length // 2)

    fc_input_size = max(1, temporal_length) * max(1, in_channels)
    fc_params = 0
    for nodes in fc_nodes:
        nodes = int(nodes)
        fc_params += (fc_input_size * nodes) + nodes
        fc_params += 2 * nodes
        fc_input_size = nodes
    fc_params += (fc_input_size * int(config['num_classes'])) + int(config['num_classes'])

    return int(conv_params + fc_params)


def is_genome_valid(genome: dict, config: dict) -> bool:
    """
    Validates if a genome will produce a valid architecture.
    Checks if the convolutional layers will reduce dimensions too much.
    
    Args:
        genome: The genome to validate
        config: Configuration dictionary
        
    Returns:
        True if genome is valid, False otherwise
    """
    fixed = normalize_residual_fields(dict(genome), config)

    # Calculate expected output size after all conv layers.
    # Sequential mode pools after every convolution; residual mode pools once
    # per residual block.
    num_conv_layers = fixed['num_conv_layers']
    sequence_length = config['sequence_length']
    residual_enabled = fixed.get('residual_enabled', False)
    residual_block_size = fixed.get('residual_block_size', DEFAULT_RESIDUAL_BLOCK_SIZE)
    pool_count = calculate_pooling_operation_count(
        num_conv_layers,
        residual_enabled=residual_enabled,
        residual_block_size=residual_block_size,
    )
    
    expected_length = sequence_length / (2 ** pool_count)
    
    # We need at least 2 values for BatchNorm to work properly
    # Use a safety margin
    min_required_length = 4
    
    if expected_length < min_required_length:
        return False
    
    # Also check that we don't have too many conv layers for the sequence length
    max_allowed_conv_layers = calculate_max_safe_conv_layers(
        sequence_length,
        min_required_length=min_required_length,
        residual_enabled=residual_enabled,
        residual_block_size=residual_block_size,
    )
    
    if num_conv_layers > max_allowed_conv_layers:
        return False

    max_model_parameters = config.get('max_model_parameters')
    if max_model_parameters is not None:
        if estimate_genome_parameter_count(fixed, config) > int(max_model_parameters):
            return False
    
    return True


def calculate_max_safe_conv_layers(
    sequence_length: int,
    min_required_length: int = 4,
    residual_enabled: bool = False,
    residual_block_size: int = DEFAULT_RESIDUAL_BLOCK_SIZE,
) -> int:
    """
    Calculates the maximum safe number of convolutional layers for a given sequence length.
    
    Args:
        sequence_length: Input sequence length
        min_required_length: Minimum required spatial dimension (default: 4)
    
    Returns:
        Maximum safe number of conv layers
    """
    if sequence_length < min_required_length:
        return 0

    max_pool_count = int(np.log2(sequence_length / min_required_length))
    if residual_enabled:
        return int(max_pool_count * max(1, int(residual_block_size)))
    return max_pool_count


def validate_and_fix_genome(genome: dict, config: dict) -> dict:
    """
    Validates and fixes a genome to ensure all lists match their corresponding layer counts.
    This prevents IndexError when building the model.
    
    Args:
        genome: The genome to validate
        config: Configuration dictionary with min/max values
    
    Returns:
        Fixed genome with correct list lengths
    """
    # Fix filters and kernel_sizes to match num_conv_layers
    num_conv = genome['num_conv_layers']
    
    # Fix filters list
    if len(genome['filters']) != num_conv:
        genome['filters'] = genome['filters'][:num_conv]
        while len(genome['filters']) < num_conv:
            genome['filters'].append(
                random.randint(config['min_filters'], config['max_filters'])
            )
    
    # Fix kernel_sizes list
    if len(genome['kernel_sizes']) != num_conv:
        genome['kernel_sizes'] = genome['kernel_sizes'][:num_conv]
        while len(genome['kernel_sizes']) < num_conv:
            genome['kernel_sizes'].append(
                random.choice(config['kernel_size_options'])
            )
    
    # Fix fc_nodes to match num_fc_layers
    num_fc = genome['num_fc_layers']
    
    if len(genome['fc_nodes']) != num_fc:
        genome['fc_nodes'] = genome['fc_nodes'][:num_fc]
        while len(genome['fc_nodes']) < num_fc:
            genome['fc_nodes'].append(
                random.randint(config['min_fc_nodes'], config['max_fc_nodes'])
            )
    
    normalize_residual_fields(genome, config)

    return genome
