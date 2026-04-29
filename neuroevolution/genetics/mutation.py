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


def _generation_delta(config: dict) -> tuple:
    """Returns a layer-count mutation delta that grows with the current generation."""
    generation_number = max(1, int(config.get('current_generation', 1)))
    generation_factor = float(config.get('structural_mutation_generation_factor', 0.5))
    max_delta = max(1, int(generation_number * generation_factor))
    return random.randint(1, max_delta), generation_number, max_delta


def _bounded_layer_target(current_value: int, delta: int, minimum: int, maximum: int) -> tuple:
    """Chooses +/- delta and clamps the result inside the configured layer bounds."""
    direction = random.choice([-1, 1])

    for candidate_direction in (direction, -direction):
        raw_target = current_value + (candidate_direction * delta)
        target = max(minimum, min(maximum, raw_target))
        if target != current_value:
            return target, candidate_direction

    return current_value, 0


def _mutate_layer_count(mutated_genome: dict, config: dict, safe_max_conv: int, fc_cap: int) -> None:
    """Mutates either Conv1D or FC depth by a generation-scaled random delta."""
    delta, generation_number, max_delta = _generation_delta(config)

    layer_kinds = ['conv', 'fc']
    random.shuffle(layer_kinds)

    for layer_kind in layer_kinds:
        if layer_kind == 'conv':
            key = 'num_conv_layers'
            minimum = int(config['min_conv_layers'])
            maximum = int(safe_max_conv)
            event_type = 'mutate_conv_layer_count'
        else:
            key = 'num_fc_layers'
            minimum = int(config['min_fc_layers'])
            maximum = int(fc_cap)
            event_type = 'mutate_fc_layer_count'

        current_value = int(mutated_genome[key])
        target, direction = _bounded_layer_target(current_value, delta, minimum, maximum)
        if target == current_value:
            continue

        mutated_genome[key] = target
        append_structural_event(
            mutated_genome,
            event_type,
            {
                'old': current_value,
                'new': target,
                'requested_delta': int(delta),
                'direction': '+' if direction > 0 else '-',
                'current_generation': int(generation_number),
                'max_delta_for_generation': int(max_delta),
                'min_layers': int(minimum),
                'max_layers': int(maximum)
            }
        )
        return


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
        for cached_key in ('skip_next_evaluation', 'cached_from_generation', 'metrics', 'evaluation_status'):
            mutated_genome.pop(cached_key, None)

        mutation_rate = config['current_mutation_rate']

        # Incremental caps controlled by current generation stage
        conv_cap = min(config.get('current_max_conv_layers', config['max_conv_layers']), config['max_conv_layers'])
        fc_cap = min(config.get('current_max_fc_layers', config['max_fc_layers']), config['max_fc_layers'])

        # Also enforce architecture safety with sequence length
        sequence_length = config['sequence_length']
        min_required_length = 4
        max_safe_conv_layers = int(np.log2(sequence_length / min_required_length))
        safe_max_conv = max(config['min_conv_layers'], min(conv_cap, max_safe_conv_layers))
        fc_cap = max(config['min_fc_layers'], fc_cap)

        # Structural mutation: jump +/- a generation-scaled amount, then clamp to safe config bounds.
        if random.random() < mutation_rate:
            _mutate_layer_count(mutated_genome, config, safe_max_conv, fc_cap)

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
    for cached_key in ('skip_next_evaluation', 'cached_from_generation', 'metrics', 'evaluation_status'):
        safe_genome.pop(cached_key, None)

    safe_genome['num_conv_layers'] = min(safe_genome['num_conv_layers'], safe_max_conv)
    safe_genome['num_fc_layers'] = min(safe_genome['num_fc_layers'], fc_cap)
    safe_genome = validate_and_fix_genome(safe_genome, config)
    safe_genome['id'] = str(uuid.uuid4())[:8]
    safe_genome['fitness'] = 0.0
    safe_genome['innovation_genes'] = build_innovation_genes(safe_genome)

    return safe_genome
