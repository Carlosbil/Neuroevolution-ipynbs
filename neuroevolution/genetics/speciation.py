"""
Speciation functions for NEAT-like evolution.
"""

import numpy as np
from typing import Dict, List


def _active_topology(genome: dict) -> str:
    """Returns the active Conv1D topology name for distance calculations."""
    if genome.get('inception_enabled', False):
        return 'inception'
    if genome.get('residual_enabled', False):
        return 'residual'
    return 'sequential'


def _conv_topology_distance(genome1: dict, genome2: dict, config: dict) -> float:
    """Returns a normalized distance for sequential/residual/Inception topology."""
    topology1 = _active_topology(genome1)
    topology2 = _active_topology(genome2)
    topology_distance = 0.0 if topology1 == topology2 else 1.0

    block_options = config.get('residual_block_size_options', [2, 3])
    block_span = max(1, max(block_options) - min(block_options)) if block_options else 1
    block_distance = abs(
        int(genome1.get('residual_block_size', 2)) - int(genome2.get('residual_block_size', 2))
    ) / block_span

    projection_distance = 0.0 if str(genome1.get('residual_projection', 'auto')) == str(genome2.get('residual_projection', 'auto')) else 1.0
    residual_distance = (min(1.0, block_distance) + projection_distance) / 2.0 if topology1 == topology2 == 'residual' else 0.0

    reduction_options = config.get('inception_reduction_ratio_options', [0.25, 0.5])
    reduction_span = max(1e-8, max(reduction_options) - min(reduction_options)) if reduction_options else 1.0
    reduction_distance = abs(
        float(genome1.get('inception_reduction_ratio', 0.5)) - float(genome2.get('inception_reduction_ratio', 0.5))
    ) / reduction_span
    pool_distance = 0.0 if bool(genome1.get('inception_pool_branch', True)) == bool(genome2.get('inception_pool_branch', True)) else 1.0
    inception_distance = (min(1.0, reduction_distance) + pool_distance) / 2.0 if topology1 == topology2 == 'inception' else 0.0

    return (topology_distance + residual_distance + inception_distance) / 3.0


def calculate_compatibility_distance(genome1: dict, genome2: dict, config: dict) -> float:
    """
    Calculates compatibility distance between two genomes.
    Combines topology differences, innovation mismatch, and hyperparameter differences.
    
    Args:
        genome1: First genome
        genome2: Second genome
        config: Configuration dictionary
    
    Returns:
        Compatibility distance (0 = identical, 1+ = very different)
    """
    # Topology distance
    topo = (
        abs(genome1['num_conv_layers'] - genome2['num_conv_layers']) +
        abs(genome1['num_fc_layers'] - genome2['num_fc_layers'])
    ) / max(1, config['max_conv_layers'] + config['max_fc_layers'])

    # Innovation gene alignment
    ids1 = {gene['innovation_id'] for gene in genome1.get('innovation_genes', [])}
    ids2 = {gene['innovation_id'] for gene in genome2.get('innovation_genes', [])}
    union_size = len(ids1 | ids2)
    innovation_mismatch = 0.0 if union_size == 0 else 1.0 - (len(ids1 & ids2) / union_size)

    # Numeric hyperparameter distance
    numeric = (
        abs(genome1.get('dropout_rate', 0.0) - genome2.get('dropout_rate', 0.0)) +
        abs(np.log10(genome1.get('learning_rate', 1e-4)) - np.log10(genome2.get('learning_rate', 1e-4))) / 4.0
    ) / 2.0

    conv_topology = _conv_topology_distance(genome1, genome2, config)

    return 0.40 * topo + 0.40 * innovation_mismatch + 0.10 * conv_topology + 0.10 * numeric


def assign_species(population: List[dict], species_dict: Dict, config: dict) -> Dict:
    """
    Assigns genomes to species based on compatibility distance.
    
    Args:
        population: List of genomes to assign
        species_dict: Current species dictionary (may be empty)
        config: Configuration dictionary
    
    Returns:
        Updated species dictionary
    """
    threshold = config.get('speciation_threshold', 0.45)
    new_species = {}
    
    for genome in population:
        # Ensure genome has innovation genes
        from neuroevolution.genetics.innovation import build_innovation_genes
        if 'innovation_genes' not in genome:
            genome['innovation_genes'] = build_innovation_genes(genome)
        
        assigned = False
        
        # Try to assign to existing species
        for species_id, specie in species_dict.items():
            distance = calculate_compatibility_distance(genome, specie['representative'], config)
            if distance <= threshold:
                if species_id not in new_species:
                    new_species[species_id] = {
                        'representative': specie['representative'],
                        'members': []
                    }
                new_species[species_id]['members'].append(genome)
                genome['species_id'] = species_id
                assigned = True
                break
        
        # Create new species if not assigned
        if not assigned:
            species_id = f"S{len(new_species) + 1}"
            new_species[species_id] = {
                'representative': genome,
                'members': [genome]
            }
            genome['species_id'] = species_id
    
    return new_species


def update_species_representatives(species_dict: Dict) -> None:
    """
    Updates species representatives to the fittest member of each species.
    
    Args:
        species_dict: Species dictionary (modified in-place)
    """
    for species_id, specie in species_dict.items():
        if specie['members']:
            # Choose fittest member as new representative
            fittest = max(specie['members'], key=lambda g: g.get('fitness', 0.0))
            specie['representative'] = fittest


def calculate_species_adjusted_fitness(species_dict: Dict) -> None:
    """
    Calculates adjusted fitness for each genome based on species size (fitness sharing).
    
    Args:
        species_dict: Species dictionary (modified in-place)
    """
    for species_id, specie in species_dict.items():
        species_size = len(specie['members'])
        if species_size > 0:
            for genome in specie['members']:
                # Fitness sharing: divide by species size to prevent single species dominance
                genome['adjusted_fitness'] = genome.get('fitness', 0.0) / species_size
