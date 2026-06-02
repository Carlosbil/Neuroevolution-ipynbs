"""
Selection operators for genetic algorithms.
"""

import numpy as np
import random
import copy
import math


DEFAULT_PARETO_OBJECTIVES = [
    {'name': 'fitness', 'direction': 'maximize'},
    {'name': 'evaluation_time_seconds', 'direction': 'minimize'},
    {'name': 'parameter_count', 'direction': 'minimize'},
]


def _normalize_strategy(config: dict) -> str:
    """Returns the configured selection strategy."""
    return str(config.get('selection_strategy', 'fitness')).lower()


def _normalize_objectives(objectives: list = None) -> list:
    """Returns normalized Pareto objective dictionaries."""
    if objectives is None:
        objectives = DEFAULT_PARETO_OBJECTIVES
    return [
        {
            'name': str(objective['name']),
            'direction': str(objective['direction']).lower(),
        }
        for objective in objectives
    ]


def _missing_objective_value(direction: str) -> float:
    """Treats missing objective values as worst possible during ranking."""
    if direction == 'maximize':
        return -math.inf
    return math.inf


def _objective_value(individual: dict, objective: dict) -> float:
    """Returns a scalar objective value, using worst-case values when missing."""
    name = objective['name']
    direction = objective['direction']
    value = individual.get(name)
    if value is None:
        metrics = individual.get('metrics')
        if isinstance(metrics, dict):
            value = metrics.get(name)
    if value is None:
        selection_objectives = individual.get('selection_objectives')
        if isinstance(selection_objectives, dict):
            value = selection_objectives.get(name)
    if value is None:
        return _missing_objective_value(direction)
    return float(value)


def _objective_tolerance(objective: dict, epsilon: float) -> float:
    """Applies epsilon only to the primary fitness objective."""
    if objective['name'] == 'fitness':
        return float(epsilon)
    return 0.0


def dominates(a: dict, b: dict, objectives: list, epsilon: float = 0.0) -> bool:
    """
    Returns True when individual a Pareto-dominates individual b.

    Missing objective values are treated as worst possible: -inf for maximize
    objectives and +inf for minimize objectives.
    """
    no_worse = True
    strictly_better = False

    for objective in _normalize_objectives(objectives):
        direction = objective['direction']
        a_value = _objective_value(a, objective)
        b_value = _objective_value(b, objective)
        tolerance = _objective_tolerance(objective, epsilon)

        if direction == 'maximize':
            if a_value < b_value - tolerance:
                no_worse = False
                break
            if a_value > b_value + tolerance:
                strictly_better = True
        elif direction == 'minimize':
            if a_value > b_value + tolerance:
                no_worse = False
                break
            if a_value < b_value - tolerance:
                strictly_better = True
        else:
            raise ValueError(f"Unsupported objective direction: {direction}")

    return no_worse and strictly_better


def non_dominated_sort(population: list, objectives: list, epsilon: float = 0.0) -> list:
    """Splits population into Pareto fronts, preserving input order inside fronts."""
    individuals = list(population)
    if not individuals:
        return []

    dominated_indices = {index: [] for index in range(len(individuals))}
    domination_counts = {index: 0 for index in range(len(individuals))}

    for p_index, p_individual in enumerate(individuals):
        for q_index, q_individual in enumerate(individuals):
            if p_index == q_index:
                continue
            if dominates(p_individual, q_individual, objectives, epsilon):
                dominated_indices[p_index].append(q_index)
            elif dominates(q_individual, p_individual, objectives, epsilon):
                domination_counts[p_index] += 1

    current_front = [
        index
        for index, count in domination_counts.items()
        if count == 0
    ]
    fronts = []

    while current_front:
        fronts.append([individuals[index] for index in current_front])
        next_front = []
        for p_index in current_front:
            for q_index in dominated_indices[p_index]:
                domination_counts[q_index] -= 1
                if domination_counts[q_index] == 0:
                    next_front.append(q_index)
        current_front = next_front

    return fronts


def _assign_crowding_distance(front: list, objectives: list) -> None:
    """Assigns NSGA-II crowding distance to one Pareto front in place."""
    if not front:
        return

    for individual in front:
        individual['crowding_distance'] = 0.0

    if len(front) <= 2:
        for individual in front:
            individual['crowding_distance'] = math.inf
        return

    for objective in _normalize_objectives(objectives):
        sorted_front = sorted(front, key=lambda individual: _objective_value(individual, objective))
        sorted_front[0]['crowding_distance'] = math.inf
        sorted_front[-1]['crowding_distance'] = math.inf

        min_value = _objective_value(sorted_front[0], objective)
        max_value = _objective_value(sorted_front[-1], objective)
        if not math.isfinite(min_value) or not math.isfinite(max_value) or max_value == min_value:
            continue

        denominator = max_value - min_value
        for index in range(1, len(sorted_front) - 1):
            if math.isinf(sorted_front[index]['crowding_distance']):
                continue
            previous_value = _objective_value(sorted_front[index - 1], objective)
            next_value = _objective_value(sorted_front[index + 1], objective)
            sorted_front[index]['crowding_distance'] += abs(next_value - previous_value) / denominator


def assign_pareto_metadata(population: list, objectives: list, epsilon: float = 0.0) -> list:
    """
    Assigns Pareto rank, crowding distance, and objective metadata in place.

    Returns:
        The computed list of fronts.
    """
    normalized_objectives = _normalize_objectives(objectives)
    fronts = non_dominated_sort(population, normalized_objectives, epsilon)

    for rank, front in enumerate(fronts):
        for individual in front:
            individual['pareto_rank'] = rank
            individual['selection_objectives'] = {
                objective['name']: None
                if _objective_value(individual, objective) in {-math.inf, math.inf}
                else _objective_value(individual, objective)
                for objective in normalized_objectives
            }
        _assign_crowding_distance(front, normalized_objectives)

    return fronts


def pareto_selection_key(individual: dict, config: dict) -> tuple:
    """
    Returns an ascending sort key for configured selection behavior.

    Pareto mode prefers lower rank, higher crowding distance, then higher
    scalar fitness. Fitness mode preserves scalar max-fitness ordering.
    """
    if _normalize_strategy(config) != 'pareto':
        return (-float(individual.get('fitness', 0.0)),)

    rank = individual.get('pareto_rank')
    if rank is None:
        rank = math.inf
    crowding_distance = individual.get('crowding_distance')
    if crowding_distance is None:
        crowding_distance = 0.0
    fitness = float(individual.get('fitness', 0.0))
    return (rank, -float(crowding_distance), -fitness)


def tournament_selection(population: list, config: dict, tournament_size: int = 3) -> dict:
    """Selects the best individual from a random tournament."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    if _normalize_strategy(config) == 'pareto':
        objectives = config.get('pareto_objectives', DEFAULT_PARETO_OBJECTIVES)
        epsilon = float(config.get('pareto_fitness_epsilon', 0.0))
        if any(individual.get('pareto_rank') is None for individual in population):
            assign_pareto_metadata(population, objectives, epsilon)
        return min(tournament, key=lambda individual: pareto_selection_key(individual, config))
    return max(tournament, key=lambda individual: individual.get('fitness', 0.0))


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
