"""
Selection operators for genetic algorithms.
"""

import numpy as np
import random
import copy


def calculate_selection_probabilities(population: list) -> np.ndarray:
    """
    Calculates fitness-proportional selection probabilities.
    
    Args:
        population: List of genomes with 'fitness' values
    
    Returns:
        Array of selection probabilities
    """
    fitnesses = np.array([ind['fitness'] for ind in population])
    
    # Handle zero or negative fitness by shifting
    min_fitness = fitnesses.min()
    if min_fitness <= 0:
        fitnesses = fitnesses - min_fitness + 1e-6
    
    # Fitness-proportional probabilities
    total_fitness = fitnesses.sum()
    if total_fitness == 0:
        # Uniform probabilities if all fitnesses are zero
        return np.ones(len(population)) / len(population)
    
    return fitnesses / total_fitness


def select_population(population: list, config: dict) -> list:
    """
    Performs selection using elitism and fitness-proportional selection.
    
    Args:
        population: Current population
        config: Configuration dictionary
    
    Returns:
        Selected population (same size as input)
    """
    population_size = len(population)
    elite_size = int(population_size * config['elite_percentage'])
    
    # Sort by fitness (descending)
    sorted_population = sorted(population, key=lambda x: x['fitness'], reverse=True)
    
    # Elite individuals (best performers)
    elite = [copy.deepcopy(ind) for ind in sorted_population[:elite_size]]
    
    # Fill rest with fitness-proportional selection
    remaining_size = population_size - elite_size
    
    if remaining_size > 0:
        probabilities = calculate_selection_probabilities(sorted_population)
        selected_indices = np.random.choice(
            len(sorted_population),
            size=remaining_size,
            replace=True,
            p=probabilities
        )
        selected = [copy.deepcopy(sorted_population[i]) for i in selected_indices]
    else:
        selected = []
    
    return elite + selected
