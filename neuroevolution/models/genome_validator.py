"""
Genome validation utilities for architecture safety checks.
"""

import numpy as np
import random


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
    # Calculate expected output size after all conv layers
    # Each MaxPool layer reduces by factor of 2
    num_conv_layers = genome['num_conv_layers']
    sequence_length = config['sequence_length']
    
    # Each conv layer has a MaxPool that divides by 2
    expected_length = sequence_length / (2 ** num_conv_layers)
    
    # We need at least 2 values for BatchNorm to work properly
    # Use a safety margin
    min_required_length = 4
    
    if expected_length < min_required_length:
        return False
    
    # Also check that we don't have too many conv layers for the sequence length
    max_allowed_conv_layers = int(np.log2(sequence_length / min_required_length))
    
    if num_conv_layers > max_allowed_conv_layers:
        return False
    
    return True


def calculate_max_safe_conv_layers(sequence_length: int, min_required_length: int = 4) -> int:
    """
    Calculates the maximum safe number of convolutional layers for a given sequence length.
    
    Args:
        sequence_length: Input sequence length
        min_required_length: Minimum required spatial dimension (default: 4)
    
    Returns:
        Maximum safe number of conv layers
    """
    return int(np.log2(sequence_length / min_required_length))


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
    
    return genome
