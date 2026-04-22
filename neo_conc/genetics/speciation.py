"""
Speciation functions for NEAT-like evolution.
"""

import numpy as np
from typing import Dict, List


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

    return 0.45 * topo + 0.45 * innovation_mismatch + 0.10 * numeric


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
        from neo_conc.genetics.innovation import build_innovation_genes
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
