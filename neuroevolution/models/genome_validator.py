"""
Genome validation utilities for architecture safety checks.
"""

import math
import numpy as np
import random


DEFAULT_RESIDUAL_BLOCK_SIZE = 2
DEFAULT_RESIDUAL_PROJECTION = "auto"
DEFAULT_CONV_TOPOLOGY = "sequential"
DEFAULT_INCEPTION_REDUCTION_RATIO = 0.5
DEFAULT_INCEPTION_POOL_BRANCH = True
DEFAULT_INCEPTION_MIN_BRANCH_CHANNELS = 1
RANDOM_TEMPLATE_ORIGIN = "random"
INITIAL_TEMPLATE_ORIGIN = "initial_seed"
MUTATION_TEMPLATE_ORIGIN = "mutation_module"


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


def _get_inception_reduction_options(config: dict) -> list:
    """Returns supported Inception reduction ratios."""
    options = config.get('inception_reduction_ratio_options', [DEFAULT_INCEPTION_REDUCTION_RATIO])
    normalized = sorted({float(option) for option in options if 0 < float(option) <= 1})
    return normalized or [DEFAULT_INCEPTION_REDUCTION_RATIO]


def _get_inception_pool_branch_options(config: dict) -> list:
    """Returns supported Inception pool-branch choices."""
    options = config.get('inception_pool_branch_options', [DEFAULT_INCEPTION_POOL_BRANCH])
    normalized = []
    for option in options:
        bool_option = _as_bool(option)
        if bool_option not in normalized:
            normalized.append(bool_option)
    return normalized or [DEFAULT_INCEPTION_POOL_BRANCH]


def _as_bool(value) -> bool:
    """Converts common persisted boolean values without treating 'False' as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _normalize_odd_kernel(kernel_size: int, minimum: int = 3) -> int:
    """Returns an odd kernel size that is at least the requested minimum."""
    kernel_size = max(int(kernel_size), int(minimum))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def calculate_inception_branch_channels(
    total_channels: int,
    pool_branch: bool = DEFAULT_INCEPTION_POOL_BRANCH,
    min_branch_channels: int = DEFAULT_INCEPTION_MIN_BRANCH_CHANNELS,
) -> dict:
    """
    Splits an Inception module's output channels across active branches.

    The split is deterministic and always sums to total_channels. A ValueError
    is raised when the total cannot satisfy the configured branch minimum.
    """
    total_channels = int(total_channels)
    min_branch_channels = max(1, int(min_branch_channels))
    branches = ['pointwise', 'medium', 'wide']
    if _as_bool(pool_branch):
        branches.append('pool')

    minimum_total = len(branches) * min_branch_channels
    if total_channels < minimum_total:
        raise ValueError(
            f"Inception module needs at least {minimum_total} channels "
            f"for {len(branches)} active branches"
        )

    allocation = {branch: min_branch_channels for branch in branches}
    remaining = total_channels - minimum_total
    branch_index = 0
    while remaining > 0:
        allocation[branches[branch_index % len(branches)]] += 1
        remaining -= 1
        branch_index += 1
    return allocation


def calculate_inception_reduction_channels(
    in_channels: int,
    reduction_ratio: float = DEFAULT_INCEPTION_REDUCTION_RATIO,
    min_branch_channels: int = DEFAULT_INCEPTION_MIN_BRANCH_CHANNELS,
) -> int:
    """Returns the 1x1 reduction width for medium and wide Inception branches."""
    return max(max(1, int(min_branch_channels)), int(round(int(in_channels) * float(reduction_ratio))))


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


def normalize_inception_fields(genome: dict, config: dict) -> dict:
    """
    Adds and normalizes Inception topology fields in-place.

    Inception kernels use the existing genome kernel list as the wide branch
    kernels and are normalized to odd values of at least 5 when active.
    """
    reduction_options = _get_inception_reduction_options(config)
    pool_options = _get_inception_pool_branch_options(config)

    inception_enabled = _as_bool(genome.get('inception_enabled', False))
    reduction_ratio = float(genome.get('inception_reduction_ratio', DEFAULT_INCEPTION_REDUCTION_RATIO))
    pool_branch = _as_bool(genome.get('inception_pool_branch', DEFAULT_INCEPTION_POOL_BRANCH))

    if reduction_ratio not in reduction_options:
        reduction_ratio = (
            DEFAULT_INCEPTION_REDUCTION_RATIO
            if DEFAULT_INCEPTION_REDUCTION_RATIO in reduction_options
            else reduction_options[0]
        )
    if pool_branch not in pool_options:
        pool_branch = (
            DEFAULT_INCEPTION_POOL_BRANCH
            if DEFAULT_INCEPTION_POOL_BRANCH in pool_options
            else pool_options[0]
        )

    genome['inception_enabled'] = inception_enabled
    genome['inception_reduction_ratio'] = reduction_ratio
    genome['inception_pool_branch'] = pool_branch
    return genome


def normalize_conv_topology_fields(genome: dict, config: dict) -> dict:
    """Normalizes sequential, residual, and Inception topology into one active mode."""
    requested_topology = str(genome.get('conv_topology', '')).lower()
    if requested_topology == 'residual':
        genome['residual_enabled'] = True
        genome['inception_enabled'] = False
    elif requested_topology == 'inception':
        genome['inception_enabled'] = True
        genome['residual_enabled'] = False

    normalize_residual_fields(genome, config)
    normalize_inception_fields(genome, config)

    if requested_topology not in {'sequential', 'residual', 'inception'}:
        if genome.get('inception_enabled', False):
            requested_topology = 'inception'
        elif genome.get('residual_enabled', False):
            requested_topology = 'residual'
        else:
            requested_topology = DEFAULT_CONV_TOPOLOGY

    if requested_topology == 'inception' and genome.get('inception_enabled', False):
        genome['residual_enabled'] = False
        genome['conv_topology'] = 'inception'
    elif requested_topology == 'residual' and genome.get('residual_enabled', False):
        genome['inception_enabled'] = False
        genome['conv_topology'] = 'residual'
    elif genome.get('inception_enabled', False):
        genome['residual_enabled'] = False
        genome['conv_topology'] = 'inception'
    elif genome.get('residual_enabled', False):
        genome['inception_enabled'] = False
        genome['conv_topology'] = 'residual'
    else:
        genome['residual_enabled'] = False
        genome['inception_enabled'] = False
        genome['conv_topology'] = DEFAULT_CONV_TOPOLOGY

    if genome['conv_topology'] == 'inception':
        min_branch_channels = int(config.get('inception_min_branch_channels', DEFAULT_INCEPTION_MIN_BRANCH_CHANNELS))
        active_branch_count = 4 if genome.get('inception_pool_branch', True) else 3
        minimum_total_channels = active_branch_count * max(1, min_branch_channels)
        max_filters = int(config.get('max_filters', minimum_total_channels))

        for index, current_filters in enumerate(genome.get('filters', [])):
            if int(current_filters) < minimum_total_channels and minimum_total_channels <= max_filters:
                genome['filters'][index] = minimum_total_channels

        for index, kernel_size in enumerate(genome.get('kernel_sizes', [])):
            genome['kernel_sizes'][index] = _normalize_odd_kernel(kernel_size, minimum=5)

    return genome


def normalize_template_provenance_fields(genome: dict) -> dict:
    """Normalizes optional architecture-template provenance fields in-place."""
    template_id = genome.get('architecture_template_id')
    template_family = genome.get('architecture_template_family')
    origin = genome.get('architecture_template_origin')

    if not template_id and not template_family:
        if origin in (None, RANDOM_TEMPLATE_ORIGIN):
            genome.pop('architecture_template_id', None)
            genome.pop('architecture_template_family', None)
            genome.pop('architecture_template_origin', None)
        return genome

    if origin not in {INITIAL_TEMPLATE_ORIGIN, MUTATION_TEMPLATE_ORIGIN}:
        genome['architecture_template_origin'] = RANDOM_TEMPLATE_ORIGIN
    return genome


def calculate_pooling_operation_count(
    num_conv_layers: int,
    residual_enabled: bool = False,
    residual_block_size: int = DEFAULT_RESIDUAL_BLOCK_SIZE,
    inception_enabled: bool = False,
) -> int:
    """Returns how many MaxPool1d operations a conv stack applies."""
    num_conv_layers = max(0, int(num_conv_layers))
    residual_block_size = max(1, int(residual_block_size))
    if residual_enabled:
        return int(math.ceil(num_conv_layers / residual_block_size))
    if inception_enabled:
        return num_conv_layers
    return num_conv_layers


def estimate_genome_parameter_count(genome: dict, config: dict) -> int:
    """
    Deterministically estimates trainable parameters without instantiating a model.
    """
    fixed = normalize_conv_topology_fields(
        {k: list(v) if isinstance(v, list) else v for k, v in genome.items()},
        config,
    )
    num_conv_layers = int(fixed['num_conv_layers'])
    filters = list(fixed.get('filters', []))[:num_conv_layers]
    kernel_sizes = list(fixed.get('kernel_sizes', []))[:num_conv_layers]
    fc_nodes = list(fixed.get('fc_nodes', []))[:int(fixed.get('num_fc_layers', 0))]

    in_channels = int(config['num_channels'])
    conv_params = 0
    block_start_channels = in_channels
    residual_enabled = bool(fixed.get('residual_enabled', False))
    inception_enabled = bool(fixed.get('inception_enabled', False))
    residual_block_size = int(fixed.get('residual_block_size', DEFAULT_RESIDUAL_BLOCK_SIZE))
    inception_reduction_ratio = float(fixed.get('inception_reduction_ratio', DEFAULT_INCEPTION_REDUCTION_RATIO))
    inception_pool_branch = bool(fixed.get('inception_pool_branch', DEFAULT_INCEPTION_POOL_BRANCH))
    min_branch_channels = int(config.get('inception_min_branch_channels', DEFAULT_INCEPTION_MIN_BRANCH_CHANNELS))

    def conv1d_unit_params(unit_in_channels: int, unit_out_channels: int, kernel_size: int) -> int:
        return (
            (int(unit_in_channels) * int(unit_out_channels) * int(kernel_size))
            + int(unit_out_channels)
            + (2 * int(unit_out_channels))
        )

    if inception_enabled:
        for index, out_channels in enumerate(filters):
            out_channels = int(out_channels)
            wide_kernel_size = _normalize_odd_kernel(
                int(kernel_sizes[index]) if index < len(kernel_sizes) else 5,
                minimum=5,
            )
            branch_channels = calculate_inception_branch_channels(
                out_channels,
                pool_branch=inception_pool_branch,
                min_branch_channels=min_branch_channels,
            )
            reduction_channels = calculate_inception_reduction_channels(
                in_channels,
                inception_reduction_ratio,
                min_branch_channels=min_branch_channels,
            )

            conv_params += conv1d_unit_params(in_channels, branch_channels['pointwise'], 1)
            conv_params += conv1d_unit_params(in_channels, reduction_channels, 1)
            conv_params += conv1d_unit_params(reduction_channels, branch_channels['medium'], 3)
            conv_params += conv1d_unit_params(in_channels, reduction_channels, 1)
            conv_params += conv1d_unit_params(reduction_channels, branch_channels['wide'], wide_kernel_size)
            if inception_pool_branch:
                conv_params += conv1d_unit_params(in_channels, branch_channels['pool'], 1)

            in_channels = out_channels
    else:
        for index, out_channels in enumerate(filters):
            kernel_size = int(kernel_sizes[index]) if index < len(kernel_sizes) else 3
            kernel_size = _normalize_odd_kernel(kernel_size, minimum=3)
            out_channels = int(out_channels)
            conv_params += conv1d_unit_params(in_channels, out_channels, kernel_size)
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
        inception_enabled=inception_enabled,
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
    fixed = normalize_conv_topology_fields(
        {k: list(v) if isinstance(v, list) else v for k, v in genome.items()},
        config,
    )

    # Calculate expected output size after all conv layers.
    # Sequential mode pools after every convolution; residual mode pools once
    # per residual block.
    num_conv_layers = fixed['num_conv_layers']
    sequence_length = config['sequence_length']
    residual_enabled = fixed.get('residual_enabled', False)
    residual_block_size = fixed.get('residual_block_size', DEFAULT_RESIDUAL_BLOCK_SIZE)
    inception_enabled = fixed.get('inception_enabled', False)
    pool_count = calculate_pooling_operation_count(
        num_conv_layers,
        residual_enabled=residual_enabled,
        residual_block_size=residual_block_size,
        inception_enabled=inception_enabled,
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
        inception_enabled=inception_enabled,
    )
    
    if num_conv_layers > max_allowed_conv_layers:
        return False

    if inception_enabled:
        min_branch_channels = int(config.get('inception_min_branch_channels', DEFAULT_INCEPTION_MIN_BRANCH_CHANNELS))
        for out_channels in fixed.get('filters', [])[:num_conv_layers]:
            try:
                calculate_inception_branch_channels(
                    out_channels,
                    pool_branch=fixed.get('inception_pool_branch', DEFAULT_INCEPTION_POOL_BRANCH),
                    min_branch_channels=min_branch_channels,
                )
            except ValueError:
                return False

    max_model_parameters = config.get('max_model_parameters')
    if max_model_parameters is not None:
        try:
            parameter_count = estimate_genome_parameter_count(fixed, config)
        except ValueError:
            return False
        if parameter_count > int(max_model_parameters):
            return False
    
    return True


def calculate_max_safe_conv_layers(
    sequence_length: int,
    min_required_length: int = 4,
    residual_enabled: bool = False,
    residual_block_size: int = DEFAULT_RESIDUAL_BLOCK_SIZE,
    inception_enabled: bool = False,
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
    if inception_enabled:
        return max_pool_count
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
    
    normalize_conv_topology_fields(genome, config)
    normalize_template_provenance_fields(genome)

    return genome


def stable_genome_signature(genome: dict, config: dict = None) -> tuple:
    """Returns a stable architecture signature suitable for fitness caches."""
    fixed = {k: list(v) if isinstance(v, list) else v for k, v in genome.items()}
    if config is not None:
        fixed = validate_and_fix_genome(fixed, config)
    else:
        fixed.setdefault('residual_enabled', False)
        fixed.setdefault('residual_block_size', DEFAULT_RESIDUAL_BLOCK_SIZE)
        fixed.setdefault('residual_projection', DEFAULT_RESIDUAL_PROJECTION)
        fixed.setdefault('inception_enabled', False)
        fixed.setdefault('inception_reduction_ratio', DEFAULT_INCEPTION_REDUCTION_RATIO)
        fixed.setdefault('inception_pool_branch', DEFAULT_INCEPTION_POOL_BRANCH)
        fixed.setdefault('conv_topology', DEFAULT_CONV_TOPOLOGY)

    return (
        fixed['num_conv_layers'],
        tuple(fixed['filters']),
        tuple(fixed['kernel_sizes']),
        fixed['num_fc_layers'],
        tuple(fixed['fc_nodes']),
        tuple(fixed['activations']),
        fixed['dropout_rate'],
        fixed['learning_rate'],
        fixed['optimizer'],
        fixed['normalization_type'],
        fixed.get('residual_enabled', False),
        fixed.get('residual_block_size', DEFAULT_RESIDUAL_BLOCK_SIZE),
        fixed.get('residual_projection', DEFAULT_RESIDUAL_PROJECTION),
        fixed.get('inception_enabled', False),
        fixed.get('inception_reduction_ratio', DEFAULT_INCEPTION_REDUCTION_RATIO),
        fixed.get('inception_pool_branch', DEFAULT_INCEPTION_POOL_BRANCH),
        fixed.get('conv_topology', DEFAULT_CONV_TOPOLOGY),
    )
