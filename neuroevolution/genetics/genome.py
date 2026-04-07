"""
Genome creation functions for neuroevolution.
"""

import random
import uuid
import numpy as np
from neuroevolution.config import ACTIVATION_FUNCTIONS, OPTIMIZERS
from neuroevolution.models.genome_validator import is_genome_valid, validate_and_fix_genome
from neuroevolution.genetics.innovation import build_innovation_genes


def create_random_genome(config: dict) -> dict:
    """
    Creates a random genome within specified ranges (optimized for 1D audio).
    Ensures the genome will produce a valid architecture.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Valid genome dictionary
    """
    max_attempts = 100
    attempt = 0

    # Incremental caps (if not present, use full limits)
    conv_cap = config.get('current_max_conv_layers', config['max_conv_layers'])
    fc_cap = config.get('current_max_fc_layers', config['max_fc_layers'])

    while attempt < max_attempts:
        # Calculate maximum safe conv layers based on sequence length
        sequence_length = config['sequence_length']
        min_required_length = 4
        max_safe_conv_layers = int(np.log2(sequence_length / min_required_length))

        # Limit conv layers to safe maximum
        safe_max_conv = min(conv_cap, max_safe_conv_layers)

        # Number of layers
        num_conv_layers = random.randint(config['min_conv_layers'], safe_max_conv)
        num_fc_layers = random.randint(config['min_fc_layers'], fc_cap)

        # Filters for each convolutional layer (progressive increase)
        filters = []
        base_filters = random.randint(config['min_filters'], config['min_filters'] * 2)
        for i in range(num_conv_layers):
            layer_filters = min(base_filters * (2 ** i), config['max_filters'])
            filters.append(layer_filters)

        # Kernel sizes (using configured options)
        kernel_sizes = [random.choice(config['kernel_size_options']) for _ in range(num_conv_layers)]

        # Nodes in fully connected layers (progressive decrease)
        fc_nodes = []
        base_fc = random.randint(config['min_fc_nodes'], config['max_fc_nodes'])
        for i in range(num_fc_layers):
            layer_nodes = max(config['min_fc_nodes'], base_fc // (2 ** i))
            fc_nodes.append(layer_nodes)

        # Activation functions for each layer
        activations = [random.choice(list(ACTIVATION_FUNCTIONS.keys())) for _ in range(max(num_conv_layers, num_fc_layers))]

        # Other parameters (using configured ranges and options)
        dropout_rate = random.uniform(config['min_dropout'], config['max_dropout'])
        learning_rate = random.choice(config['learning_rate_options'])
        optimizer = random.choice(list(OPTIMIZERS.keys()))
        normalization_type = 'batch'

        genome = {
            'num_conv_layers': num_conv_layers,
            'num_fc_layers': num_fc_layers,
            'filters': filters,
            'kernel_sizes': kernel_sizes,
            'fc_nodes': fc_nodes,
            'activations': activations,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'normalization_type': normalization_type,
            'fitness': 0.0,
            'id': str(uuid.uuid4())[:8],
            'structural_history': []
        }

        if is_genome_valid(genome, config):
            genome['innovation_genes'] = build_innovation_genes(genome)
            return genome

        attempt += 1

    print(f"⚠️ Warning: Could not create random genome after {max_attempts} attempts. Creating minimal safe genome.")
    genome = {
        'num_conv_layers': 1,
        'num_fc_layers': 1,
        'filters': [32],
        'kernel_sizes': [3],
        'fc_nodes': [64],
        'activations': ['relu', 'relu'],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'normalization_type': 'batch',
        'fitness': 0.0,
        'id': str(uuid.uuid4())[:8],
        'structural_history': []
    }
    genome['innovation_genes'] = build_innovation_genes(genome)
    return genome
