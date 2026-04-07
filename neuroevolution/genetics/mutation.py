"""
Mutation operators for genome evolution.
"""

import random
import copy
import uuid
import numpy as np
from neuroevolution.config import ACTIVATION_FUNCTIONS, OPTIMIZERS
from neuroevolution.models.genome_validator import is_genome_valid, validate_and_fix_genome
from neuroevolution.genetics.innovation import build_innovation_genes, append_structural_event


def mutate_genome(genome: dict, config: dict) -> dict:
    """
    Applies mutation to a genome using adaptive mutation rate and configurable parameters.
    Includes incremental structural growth and innovation UUID tracking.
    
    Args:
        genome: Original genome to mutate
        config: Configuration dictionary
    
    Returns:
        Mutated genome (new copy)
    """
    max_attempts = 50

    for attempt in range(max_attempts):
        mutated_genome = copy.deepcopy(genome)
        mutation_rate = config['current_mutation_rate']

        # Incremental caps controlled by current generation stage
        conv_cap = config.get('current_max_conv_layers', config['max_conv_layers'])
        fc_cap = config.get('current_max_fc_layers', config['max_fc_layers'])

        # Also enforce architecture safety with sequence length
        sequence_length = config['sequence_length']
        min_required_length = 4
        max_safe_conv_layers = int(np.log2(sequence_length / min_required_length))
        safe_max_conv = min(conv_cap, max_safe_conv_layers)

        # Structural mutation: mostly additive growth to avoid early over-complexity
        if random.random() < mutation_rate:
            grow_probability = config.get('incremental_growth_probability', 0.7)
            if random.random() < grow_probability:
                if random.random() < 0.5 and mutated_genome['num_conv_layers'] < safe_max_conv:
                    mutated_genome['num_conv_layers'] += 1
                    append_structural_event(mutated_genome, 'add_conv_layer', {'new_num_conv_layers': mutated_genome['num_conv_layers']})
                elif mutated_genome['num_fc_layers'] < fc_cap:
                    mutated_genome['num_fc_layers'] += 1
                    append_structural_event(mutated_genome, 'add_fc_layer', {'new_num_fc_layers': mutated_genome['num_fc_layers']})
            else:
                if random.random() < 0.5 and mutated_genome['num_conv_layers'] > config['min_conv_layers']:
                    mutated_genome['num_conv_layers'] -= 1
                    append_structural_event(mutated_genome, 'remove_conv_layer', {'new_num_conv_layers': mutated_genome['num_conv_layers']})
                elif mutated_genome['num_fc_layers'] > config['min_fc_layers']:
                    mutated_genome['num_fc_layers'] -= 1
                    append_structural_event(mutated_genome, 'remove_fc_layer', {'new_num_fc_layers': mutated_genome['num_fc_layers']})

        # Validate and fix lists to match layer counts
        mutated_genome = validate_and_fix_genome(mutated_genome, config)

        # Mutate filters
        for i in range(len(mutated_genome['filters'])):
            if random.random() < mutation_rate:
                old_val = mutated_genome['filters'][i]
                new_val = random.randint(config['min_filters'], config['max_filters'])
                mutated_genome['filters'][i] = new_val
                append_structural_event(mutated_genome, 'mutate_conv_filter', {'index': i, 'old': int(old_val), 'new': int(new_val)})

        # Mutate kernel sizes
        for i in range(len(mutated_genome['kernel_sizes'])):
            if random.random() < mutation_rate:
                old_val = mutated_genome['kernel_sizes'][i]
                new_val = random.choice(config['kernel_size_options'])
                mutated_genome['kernel_sizes'][i] = new_val
                append_structural_event(mutated_genome, 'mutate_conv_kernel', {'index': i, 'old': int(old_val), 'new': int(new_val)})

        # Mutate FC nodes
        for i in range(len(mutated_genome['fc_nodes'])):
            if random.random() < mutation_rate:
                old_val = mutated_genome['fc_nodes'][i]
                new_val = random.randint(config['min_fc_nodes'], config['max_fc_nodes'])
                mutated_genome['fc_nodes'][i] = new_val
                append_structural_event(mutated_genome, 'mutate_fc_node', {'index': i, 'old': int(old_val), 'new': int(new_val)})

        # Mutate activation functions
        for i in range(len(mutated_genome['activations'])):
            if random.random() < mutation_rate:
                mutated_genome['activations'][i] = random.choice(list(ACTIVATION_FUNCTIONS.keys()))

        # Mutate dropout
        if random.random() < mutation_rate:
            mutated_genome['dropout_rate'] = random.uniform(config['min_dropout'], config['max_dropout'])

        # Mutate learning rate
        if random.random() < mutation_rate:
            mutated_genome['learning_rate'] = random.choice(config['learning_rate_options'])

        # Mutate optimizer
        if random.random() < mutation_rate:
            mutated_genome['optimizer'] = random.choice(list(OPTIMIZERS.keys()))

        # Mutate normalization type
        if random.random() < mutation_rate:
            mutated_genome['normalization_type'] = random.choices(
                ['batch', 'layer'],
                weights=[config['normalization_batch_weight'], config['normalization_layer_weight']]
            )[0]

        mutated_genome['id'] = str(uuid.uuid4())[:8]
        mutated_genome['fitness'] = 0.0

        # Final validation
        mutated_genome = validate_and_fix_genome(mutated_genome, config)

        if is_genome_valid(mutated_genome, config):
            mutated_genome['innovation_genes'] = build_innovation_genes(mutated_genome)
            return mutated_genome

    print(f"⚠️ Warning: Could not create valid mutation after {max_attempts} attempts. Using safe fallback.")
    safe_genome = copy.deepcopy(genome)
    safe_genome['num_conv_layers'] = min(safe_genome['num_conv_layers'], safe_max_conv)
    safe_genome['num_fc_layers'] = min(safe_genome['num_fc_layers'], fc_cap)
    safe_genome = validate_and_fix_genome(safe_genome, config)
    safe_genome['id'] = str(uuid.uuid4())[:8]
    safe_genome['fitness'] = 0.0
    safe_genome['innovation_genes'] = build_innovation_genes(safe_genome)

    return safe_genome
