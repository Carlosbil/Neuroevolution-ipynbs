"""
Evolution engine module - main HybridNeuroevolution class.

This module implements the core evolution loop with:
- Population initialization with incremental complexity growth
- Genetic speciation based on compatibility distance
- Adaptive mutation rates based on population diversity
- Elitism and fitness-proportional selection
- Generation-level early stopping
- Checkpoint management and progress tracking
"""

import os
import copy
import random
import uuid
import json
import time
import gc
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Empty, SimpleQueue
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from ..config import get_data_phase_config, resolve_config, get_operator_config, with_fold_progress_callback
from ..genetics.genome import create_random_genome
from ..genetics.architecture_templates import (
    choose_template,
    create_template_genome,
    get_configured_templates,
)
from ..genetics.mutation import mutate_genome
from ..genetics.crossover import crossover_genomes
from ..genetics.selection import (
    DEFAULT_PARETO_OBJECTIVES,
    assign_pareto_metadata,
    pareto_selection_key,
    tournament_selection as select_tournament,
)
from ..genetics.innovation import build_innovation_genes, append_structural_event
from ..models.architecture_formatting import format_genome_architecture
from ..models.genome_validator import (
    architecture_spatial_limit,
    calculate_max_safe_conv_layers,
    estimate_genome_parameter_count,
    validate_and_fix_genome,
)
from ..models.evolvable_cnn import EvolvableCNN
from .fitness import aggregate_fold_results, train_fold_in_thread, train_fold_on_cuda_stream
from .fold_scheduler import FoldMemoryFailure, VramFoldScheduler, is_cuda_memory_error


class HybridNeuroevolution:
    """Main class that implements hybrid neuroevolution with 5-fold validation fitness and adaptive mutation."""

    def __init__(self, config: dict, device: torch.device):
        self.config = resolve_config(config)
        self.device = device
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        self.generation_stats = []
        self.fold_progress = {}

        self.run_artifacts_dir = self.config.get('artifacts_dir', 'artifacts/test_audio')
        os.makedirs(self.run_artifacts_dir, exist_ok=True)
        self.best_checkpoint_path = None
        self.progress_json_path = os.path.join(self.run_artifacts_dir, 'evolution_progress.json')
        self.generation_log_path = os.path.join(self.run_artifacts_dir, 'generation_progress.txt')
        self.fold_progress_path = os.path.join(self.run_artifacts_dir, 'fold_progress.json')

        # Early stopping configuration at generation level
        self.generations_without_improvement = 0
        self.best_fitness_overall = -float('inf')
        self.best_fitness_for_early_stopping = -float('inf')
        self.min_improvement_threshold = self.config['min_improvement_threshold']
        self.max_generations_without_improvement = self.config['early_stopping_generations']

        self.current_max_conv_layers = self.config['initial_max_conv_layers']
        self.current_max_fc_layers = self.config['initial_max_fc_layers']
        self.current_mutation_rate = self.config['current_mutation_rate']
        self.last_evaluated_generation = None
        self.template_cursor = 0
        self.species = {}

    def _uses_pareto_selection(self) -> bool:
        """Returns whether reproduction should use Pareto rank metadata."""
        return str(self.config.get('selection_strategy', 'fitness')).lower() == 'pareto'

    def _assign_selection_metadata(self):
        """Assigns Pareto metadata when the configured strategy requires it."""
        if not self._uses_pareto_selection():
            return

        assign_pareto_metadata(
            self.population,
            self.config.get('pareto_objectives', DEFAULT_PARETO_OBJECTIVES),
            float(self.config.get('pareto_fitness_epsilon', 0.0)),
        )

    def _record_cost_metadata(self, genome: dict, metrics: dict, elapsed_seconds: Optional[float] = None):
        """Stores individual-level cost metadata on the genome and metrics."""
        if elapsed_seconds is not None:
            genome['evaluation_time_seconds'] = float(elapsed_seconds)
        elif 'evaluation_time_seconds' not in genome:
            genome['evaluation_time_seconds'] = metrics.get('evaluation_time_seconds')
        metrics['evaluation_time_seconds'] = genome.get('evaluation_time_seconds')

        parameter_count = metrics.get('parameter_count') if elapsed_seconds is not None else genome.get('parameter_count', metrics.get('parameter_count'))
        if parameter_count is None:
            try:
                parameter_count = int(estimate_genome_parameter_count(genome, self.config))
            except Exception as e:
                parameter_count = None
                print(f"      WARNING: Could not estimate parameter count for {genome.get('id', 'unknown')}: {e}")

        genome['parameter_count'] = parameter_count
        metrics['parameter_count'] = parameter_count

    def _sync_selection_metadata_to_metrics(self, genome: dict, metrics: dict):
        """Copies selection metadata into metrics for progress artifacts."""
        for key in (
            'evaluation_time_seconds',
            'parameter_count',
            'pareto_rank',
            'crowding_distance',
            'selection_objectives',
        ):
            if key in genome:
                metrics[key] = genome[key]

    @staticmethod
    def _format_seconds(value) -> str:
        """Formats an optional elapsed-time value for logs."""
        if value is None:
            return "NA"
        return f"{float(value):.3f}"

    @staticmethod
    def _format_parameter_count(value) -> str:
        """Formats an optional parameter count for logs."""
        if value is None:
            return "NA"
        return str(int(value))

    @staticmethod
    def _format_crowding_distance(value) -> str:
        """Formats crowding distance for compact logs."""
        if value is None:
            return "NA"
        if np.isinf(float(value)):
            return "inf"
        return f"{float(value):.3f}"

    def _append_generation_log(self, text: str):
        """Appends text to generation log file."""
        with open(self.generation_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(text + "\n")

    def _to_json_serializable(self, value):
        """Converts numpy types to JSON-serializable Python types."""
        if isinstance(value, dict):
            return {k: self._to_json_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_json_serializable(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _save_evolution_progress(self):
        """Saves current evolution state to JSON file."""
        progress_data = {
            'schema_version': 2,
            'generation': self.generation,
            'population': self._to_json_serializable(self.population),
            'best_individual': self._to_json_serializable(self.best_individual),
            'fitness_history': self._to_json_serializable(self.fitness_history),
            'generation_stats': self._to_json_serializable(self.generation_stats),
            'generations_without_improvement': self.generations_without_improvement,
            'best_fitness_overall': self.best_fitness_overall,
            'best_fitness_for_early_stopping': self.best_fitness_for_early_stopping,
            'best_checkpoint_path': self.best_checkpoint_path,
            'selection_strategy': self.config.get('selection_strategy', 'fitness'),
            'pareto_objectives': self._to_json_serializable(
                self.config.get('pareto_objectives', DEFAULT_PARETO_OBJECTIVES)
            ),
            'pareto_fitness_epsilon': float(self.config.get('pareto_fitness_epsilon', 0.0)),
            'config_snapshot': self._to_json_serializable(self.config),
            'last_evaluated_generation': self.last_evaluated_generation,
            'template_cursor': self.template_cursor,
            'runtime_state': {
                'current_mutation_rate': self.current_mutation_rate,
                'current_max_conv_layers': self.current_max_conv_layers,
                'current_max_fc_layers': self.current_max_fc_layers,
            },
        }

        temporary_path = self.progress_json_path + '.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as progress_file:
            json.dump(progress_data, progress_file, indent=2)
        os.replace(temporary_path, self.progress_json_path)

    def _load_evolution_progress(self) -> bool:
        """Restore this experiment; a failed load must never reset its artifacts."""
        if not self.config['resume_evolution'] or not os.path.exists(self.progress_json_path):
            return False

        try:
            with open(self.progress_json_path, 'r', encoding='utf-8') as progress_file:
                progress_data = json.load(progress_file)

            if not isinstance(progress_data, dict):
                raise ValueError('Evolution progress must be a JSON object')
            saved_config = progress_data.get('config_snapshot')
            if not isinstance(saved_config, dict):
                raise ValueError('Legacy progress has no configuration snapshot; set resume_evolution=False to start a clean experiment')
            training_keys = ('input_modality', 'num_channels', 'num_frequency_bins', 'sequence_length',
                             'num_classes', 'batch_size', 'num_epochs', 'epoch_patience',
                             'seed', 'num_folds', 'fitness_metric', 'fold_selection_metric',
                             'improvement_threshold', 'metric_improvement_threshold',
                             'validation_frequency_epochs', 'max_batches_per_epoch',
                             'pre_fc_pool_shape', 'pre_fc_pool_length', 'conv_dropout_rate',
                             'use_amp', 'amp_dtype', 'require_all_folds', 'reference_aux_loss_weight',
                             'evolution_dataset_id', 'evolution_fold_files_subdirectory',
                             'optimizer_kwargs', 'reference_input_size', 'reference_input_sizes', 'reference_lenet_input_size',
                             'reference_inception_input_size', 'reference_aux_logits',
                             'reference_dropout_rate', 'minimum_specificity', 'minimum_sensitivity')
            current_config = json.loads(json.dumps(self.config))
            changed = [key for key in training_keys if saved_config.get(key) != current_config.get(key)]
            # Dataset directories may move with the project (including Linux ->
            # Windows). Keep checking dataset identity and the directory name.
            def portable_basename(path):
                return str(path).replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]

            if portable_basename(saved_config.get('data_path')) != portable_basename(current_config.get('data_path')):
                changed.append('data_path')
            population = progress_data.get('population')
            if not isinstance(population, list) or not all(isinstance(genome, dict) and genome.get('id') for genome in population):
                raise ValueError('Evolution progress has no valid saved population')
            if changed or len(progress_data.get('population', [])) != self.config['population_size']:
                raise ValueError(f'Resume configuration differs from saved training: {changed}; '
                                 'use the saved configuration or a different artifacts_dir')

            self.generation = int(progress_data.get('generation', 0))
            self.population = progress_data.get('population', [])
            self.best_individual = progress_data.get('best_individual', None)
            self.fitness_history = progress_data.get('fitness_history', [])
            self.generation_stats = progress_data.get('generation_stats', [])
            self.generations_without_improvement = int(progress_data.get('generations_without_improvement', 0))
            self.best_fitness_overall = float(progress_data.get('best_fitness_overall', -float('inf')))
            self.best_fitness_for_early_stopping = float(
                progress_data.get('best_fitness_for_early_stopping', self.best_fitness_overall)
            )
            self.best_checkpoint_path = progress_data.get('best_checkpoint_path', None)
            self.last_evaluated_generation = progress_data.get('last_evaluated_generation')
            self.template_cursor = int(progress_data.get('template_cursor', 0))
            self._update_incremental_complexity()
            runtime_state = progress_data.get('runtime_state', {})
            self.current_mutation_rate = runtime_state.get('current_mutation_rate', self.current_mutation_rate)
            self.current_max_conv_layers = runtime_state.get('current_max_conv_layers', self.current_max_conv_layers)
            self.current_max_fc_layers = runtime_state.get('current_max_fc_layers', self.current_max_fc_layers)
            if self.best_checkpoint_path:
                local_checkpoint = os.path.join(self.run_artifacts_dir, portable_basename(self.best_checkpoint_path))
                if os.path.isfile(local_checkpoint):
                    self.best_checkpoint_path = local_checkpoint
                elif not os.path.isfile(self.best_checkpoint_path):
                    print(f"WARNING: Saved best model weights are missing: {local_checkpoint}. "
                          'Resuming population and history from JSON without the previous model weights.')
                    self.best_checkpoint_path = None
            return True
        except Exception as e:
            raise ValueError(f"Could not resume evolution from {self.progress_json_path}: {e}. "
                             'Existing artifacts have been preserved.') from e

    def _update_incremental_complexity(self):
        """Updates complexity caps based on current generation."""
        step = max(1, self.config['complexity_step_generations'])
        stage = max(0, self.generation // step)

        self.current_max_conv_layers = min(
            self.config['max_conv_layers'],
            self.config['initial_max_conv_layers'] + stage
        )
        self.current_max_fc_layers = min(
            self.config['max_fc_layers'],
            self.config['initial_max_fc_layers'] + stage
        )

    def _enforce_complexity_caps(self, genome: dict) -> dict:
        """Enforces incremental complexity caps on a genome."""
        if genome.get('model_kind') == 'reference':
            return validate_and_fix_genome(genome, self.config)
        genome['num_conv_layers'] = min(genome['num_conv_layers'], self.current_max_conv_layers)
        genome['num_fc_layers'] = min(genome['num_fc_layers'], self.current_max_fc_layers)
        genome = validate_and_fix_genome(genome, self.config)
        genome['innovation_genes'] = build_innovation_genes(genome)
        return genome

    def _max_safe_conv_layers_for_genome(self, genome: dict) -> int:
        """Returns safe conv depth for a genome's current topology."""
        fixed = validate_and_fix_genome(copy.deepcopy(genome), self.config)
        return calculate_max_safe_conv_layers(
            architecture_spatial_limit(self.config),
            min_required_length=4,
            residual_enabled=fixed.get('residual_enabled', False),
            residual_block_size=fixed.get('residual_block_size', 2),
            inception_enabled=fixed.get('inception_enabled', False),
        )

    def _create_double_cap_individual(self) -> dict:
        """Creates one individual using doubled layer caps for the current generation."""
        base_conv_cap = max(
            self.config['min_conv_layers'],
            self.current_max_conv_layers
        )
        base_fc_cap = max(
            self.config['min_fc_layers'],
            self.current_max_fc_layers
        )

        target_conv_layers = min(self.config['max_conv_layers'], base_conv_cap * 2)
        target_fc_layers = min(self.config['max_fc_layers'], base_fc_cap * 2)

        target_fc_layers = max(self.config['min_fc_layers'], target_fc_layers)

        boosted_config = get_operator_config(self.config, self.generation, self.current_mutation_rate, target_conv_layers, target_fc_layers)

        genome = create_random_genome(boosted_config)
        max_safe_conv_layers = self._max_safe_conv_layers_for_genome(genome)
        target_conv_layers = max(
            self.config['min_conv_layers'],
            min(target_conv_layers, max_safe_conv_layers)
        )
        genome['num_conv_layers'] = target_conv_layers
        genome['num_fc_layers'] = target_fc_layers
        genome = validate_and_fix_genome(genome, boosted_config)
        genome['innovation_genes'] = build_innovation_genes(genome)
        genome['id'] = str(uuid.uuid4())[:8]
        genome['fitness'] = 0.0

        append_structural_event(
            genome,
            'double_cap_seed',
            {
                'base_conv_cap': int(base_conv_cap),
                'base_fc_cap': int(base_fc_cap),
                'target_conv_layers': int(genome['num_conv_layers']),
                'target_fc_layers': int(genome['num_fc_layers'])
            }
        )
        return genome

    def _template_seed_quota(self, regular_population_size: int) -> int:
        """Returns how many regular initial slots can be template-derived."""
        regular_population_size = max(0, int(regular_population_size))
        if regular_population_size <= 0:
            return 0

        seed_fraction = float(self.config.get('architecture_template_seed_fraction', 0.0))
        if seed_fraction <= 0:
            return 0

        min_random_fraction = float(self.config.get('architecture_template_seed_min_random_fraction', 0.0))
        requested_template_slots = int(np.floor(regular_population_size * seed_fraction))
        random_slots_required = int(np.ceil(regular_population_size * min_random_fraction))
        max_template_slots = max(0, regular_population_size - random_slots_required)
        return max(0, min(requested_template_slots, max_template_slots))

    @staticmethod
    def _interleaved_template_slots(regular_population_size: int, template_seed_quota: int) -> set:
        """Returns population indexes for template seeds without grouping them at the front."""
        if template_seed_quota <= 0:
            return set()
        return {
            min(
                regular_population_size - 1,
                int(((slot_index + 0.5) * regular_population_size) / template_seed_quota),
            )
            for slot_index in range(template_seed_quota)
        }

    def _create_template_seed_individual(
        self,
        quartile_config: dict,
        index_in_population: int,
        quartile_number: int,
        conv_quartile_cap: int,
        fc_quartile_cap: int,
    ) -> dict:
        """Creates one initial individual from a configured architecture template."""
        max_attempts = int(self.config.get('architecture_template_max_attempts', 20))
        last_error = None
        template = None

        for _ in range(max_attempts):
            try:
                templates = get_configured_templates(quartile_config)
                if not templates:
                    return create_random_genome(quartile_config)
                template = templates[self.template_cursor % len(templates)]
                self.template_cursor += 1
                genome = create_template_genome(
                    template.template_id,
                    quartile_config,
                    origin='initial_seed',
                    event_payload={
                        'index_in_population': int(index_in_population),
                        'quartile': int(quartile_number),
                        'conv_cap': int(conv_quartile_cap),
                        'fc_cap': int(fc_quartile_cap),
                    },
                )
                if genome.get('model_kind') == 'reference':
                    return genome
                genome_safe_max_conv_layers = self._max_safe_conv_layers_for_genome(genome)
                genome['num_conv_layers'] = max(
                    self.config['min_conv_layers'],
                    min(genome['num_conv_layers'], conv_quartile_cap, genome_safe_max_conv_layers),
                )
                genome['num_fc_layers'] = max(
                    self.config['min_fc_layers'],
                    min(genome['num_fc_layers'], fc_quartile_cap),
                )
                genome = validate_and_fix_genome(genome, quartile_config)
                genome['innovation_genes'] = build_innovation_genes(genome)
                genome['id'] = str(uuid.uuid4())[:8]
                genome['fitness'] = 0.0
                return genome
            except ValueError as error:
                if template is not None and template.model_kind == 'reference' and not self.config['reference_seed_fallback_to_random']:
                    raise
                last_error = error

        if not self.config['reference_seed_fallback_to_random']:
            raise ValueError(f"Configured template cannot be built without shrinking: {last_error}")
        print(f"WARNING: Could not create template seed after {max_attempts} attempts: {last_error}")
        return create_random_genome(quartile_config)

    def _active_topology(self, genome: dict) -> str:
        """Returns the active Conv1D topology name."""
        if genome.get('inception_enabled', False):
            return 'inception'
        if genome.get('residual_enabled', False):
            return 'residual'
        return 'sequential'

    def _conv_topology_distance(self, g1: dict, g2: dict) -> float:
        """Returns a normalized distance for sequential/residual/Inception topology."""
        topology1 = self._active_topology(g1)
        topology2 = self._active_topology(g2)
        topology_distance = 0.0 if topology1 == topology2 else 1.0

        block_options = self.config.get('residual_block_size_options', [2, 3])
        block_span = max(1, max(block_options) - min(block_options)) if block_options else 1
        block_distance = abs(
            int(g1.get('residual_block_size', 2)) - int(g2.get('residual_block_size', 2))
        ) / block_span
        projection_distance = 0.0 if str(g1.get('residual_projection', 'auto')) == str(g2.get('residual_projection', 'auto')) else 1.0
        residual_distance = (min(1.0, block_distance) + projection_distance) / 2.0 if topology1 == topology2 == 'residual' else 0.0

        reduction_options = self.config.get('inception_reduction_ratio_options', [0.25, 0.5])
        reduction_span = max(1e-8, max(reduction_options) - min(reduction_options)) if reduction_options else 1.0
        reduction_distance = abs(
            float(g1.get('inception_reduction_ratio', 0.5)) - float(g2.get('inception_reduction_ratio', 0.5))
        ) / reduction_span
        pool_distance = 0.0 if bool(g1.get('inception_pool_branch', True)) == bool(g2.get('inception_pool_branch', True)) else 1.0
        inception_distance = (min(1.0, reduction_distance) + pool_distance) / 2.0 if topology1 == topology2 == 'inception' else 0.0
        return (topology_distance + residual_distance + inception_distance) / 3.0

    def _format_architecture(self, genome: dict) -> str:
        """Formats architecture topology for logs and cached generation summaries."""
        return format_genome_architecture(genome)

    def compatibility_distance(self, g1: dict, g2: dict) -> float:
        """Combines topology differences and innovation mismatch for speciation."""
        if g1.get('model_kind') == 'reference' or g2.get('model_kind') == 'reference':
            return 0.0 if g1.get('reference_architecture_id') == g2.get('reference_architecture_id') else 1.0
        topo = (
            abs(g1['num_conv_layers'] - g2['num_conv_layers']) +
            abs(g1['num_fc_layers'] - g2['num_fc_layers'])
        ) / max(1, self.config['max_conv_layers'] + self.config['max_fc_layers'])

        ids1 = {gene['innovation_id'] for gene in g1.get('innovation_genes', [])}
        ids2 = {gene['innovation_id'] for gene in g2.get('innovation_genes', [])}
        union_size = len(ids1 | ids2)
        innovation_mismatch = 0.0 if union_size == 0 else 1.0 - (len(ids1 & ids2) / union_size)

        numeric = (
            abs(g1.get('dropout_rate', 0.0) - g2.get('dropout_rate', 0.0)) +
            abs(np.log10(g1.get('learning_rate', 1e-4)) - np.log10(g2.get('learning_rate', 1e-4))) / 4.0
        ) / 2.0

        conv_topology = self._conv_topology_distance(g1, g2)

        return 0.40 * topo + 0.40 * innovation_mismatch + 0.10 * conv_topology + 0.10 * numeric

    def _speciate_population(self):
        """Assigns genomes to species based on compatibility distance."""
        self.species = {}
        threshold = self.config['speciation_threshold']

        for genome in self.population:
            genome['innovation_genes'] = build_innovation_genes(genome)
            assigned = False

            for species_id, specie in self.species.items():
                distance = self.compatibility_distance(genome, specie['representative'])
                if distance <= threshold:
                    specie['members'].append(genome)
                    genome['species_id'] = species_id
                    assigned = True
                    break

            if not assigned:
                species_id = f"S{len(self.species) + 1}"
                self.species[species_id] = {
                    'representative': genome,
                    'members': [genome]
                }
                genome['species_id'] = species_id

    def initialize_population(self):
        """Initializes population or resumes from saved progress."""
        if self._load_evolution_progress():
            print(f"Resuming evolution from generation {self.generation} using: {self.progress_json_path}")
            return

        with open(self.generation_log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("Generation progress log\n")

        self._update_incremental_complexity()
        print(f"Initializing population of {self.config['population_size']} individuals...")
        print(
            f"Incremental caps at generation {self.generation}: "
            f"conv<={self.current_max_conv_layers}, fc<={self.current_max_fc_layers}"
        )
        print("Using mixed quartile initialization for initial population depth ranges.")

        reserve_double_cap_slot = self.config['population_size'] > 1
        regular_population_size = (
            self.config['population_size'] - 1
            if reserve_double_cap_slot
            else self.config['population_size']
        )

        self.population = []
        min_conv_layers = int(self.config['min_conv_layers'])
        min_fc_layers = int(self.config['min_fc_layers'])
        template_seed_quota = self._template_seed_quota(regular_population_size)
        template_slots = self._interleaved_template_slots(regular_population_size, template_seed_quota)
        if template_seed_quota:
            print(f"Using {template_seed_quota} known architecture template seeds.")

        for i in range(regular_population_size):
            # Quartiles: Q1 -> 25% of max depth caps, Q2 -> 50%, Q3 -> 75%, Q4 -> 100%.
            quartile_index = min(3, (4 * i) // max(1, regular_population_size))
            quartile_number = quartile_index + 1

            conv_quartile_cap = int(np.ceil((quartile_number / 4.0) * self.current_max_conv_layers))
            fc_quartile_cap = int(np.ceil((quartile_number / 4.0) * self.current_max_fc_layers))

            conv_quartile_cap = max(min_conv_layers, conv_quartile_cap)
            fc_quartile_cap = max(min_fc_layers, fc_quartile_cap)

            quartile_config = get_operator_config(self.config, self.generation, self.current_mutation_rate, conv_quartile_cap, fc_quartile_cap)

            if i in template_slots:
                genome = self._create_template_seed_individual(
                    quartile_config,
                    i,
                    quartile_number,
                    conv_quartile_cap,
                    fc_quartile_cap,
                )
                self.population.append(genome)
                continue

            genome = create_random_genome(quartile_config)
            genome_safe_max_conv_layers = self._max_safe_conv_layers_for_genome(genome)
            genome_quartile_cap = max(min_conv_layers, min(conv_quartile_cap, genome_safe_max_conv_layers))

            # Force layer counts to remain inside the quartile range requested by the user.
            genome['num_conv_layers'] = random.randint(min_conv_layers, genome_quartile_cap)
            genome['num_fc_layers'] = random.randint(min_fc_layers, fc_quartile_cap)
            genome = validate_and_fix_genome(genome, quartile_config)
            genome['innovation_genes'] = build_innovation_genes(genome)

            append_structural_event(
                genome,
                'quartile_init',
                {
                    'index_in_population': i,
                    'quartile': quartile_number,
                    'conv_cap': genome_quartile_cap,
                    'fc_cap': fc_quartile_cap,
                    'num_conv_layers': int(genome['num_conv_layers']),
                    'num_fc_layers': int(genome['num_fc_layers'])
                }
            )

            genome['id'] = str(uuid.uuid4())[:8]
            genome['fitness'] = 0.0
            self.population.append(genome)

        if reserve_double_cap_slot:
            self.population.append(self._create_double_cap_individual())

        print(f"Population initialized with {len(self.population)} individuals")

    def save_best_checkpoint(self, genome: dict, model: nn.Module):
        """
        Guarda el checkpoint del mejor modelo global y elimina el anterior.

        Args:
            genome: Genoma del mejor modelo
            model: Modelo de PyTorch a guardar
        """
        checkpoint_dir = self.run_artifacts_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_filename = f"best_model_gen{self.generation}_id{genome['id']}_fitness{genome['fitness']:.2f}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'genome': genome,
            'generation': self.generation,
            'fitness': genome['fitness'],
            'config': self.config,
            'selection_split': 'validation',
            'fitness_metric': self.config.get('fitness_metric', 'f1_score'),
            'checkpoint_metric': self.config.get('checkpoint_metric', self.config.get('fitness_metric', 'f1_score')),
            'test_evaluated': False,
        }

        previous_path = self.best_checkpoint_path
        temporary_path = checkpoint_path + '.tmp'
        try:
            torch.save(checkpoint_data, temporary_path)
            os.replace(temporary_path, checkpoint_path)
            self.best_checkpoint_path = checkpoint_path
            self._save_evolution_progress()
            if previous_path and Path(previous_path).resolve() != Path(checkpoint_path).resolve() and Path(previous_path).resolve().parent == Path(checkpoint_dir).resolve():
                try:
                    Path(previous_path).unlink(missing_ok=True)
                except OSError as error:
                    print(f"      Could not remove previous checkpoint: {error}")
            print(f"      Nuevo checkpoint guardado: {checkpoint_path}")
            print(f"        Fitness: {genome['fitness']:.2f}%, ID: {genome['id']}, Gen: {self.generation}")
        except Exception as e:
            raise RuntimeError(f"Could not save checkpoint: {checkpoint_path}") from e

    def load_best_checkpoint(self) -> tuple:
        """
        Carga el mejor checkpoint guardado.

        Returns:
            Tuple de (genome, model) o (None, None) si no hay checkpoint
        """
        if not self.best_checkpoint_path or not os.path.exists(self.best_checkpoint_path):
            print("No hay checkpoint disponible para cargar")
            return None, None

        try:
            checkpoint_data = torch.load(self.best_checkpoint_path, map_location=self.device, weights_only=False)
            genome = checkpoint_data['genome']

            model = EvolvableCNN(genome, self.config).to(self.device)
            model.load_state_dict(checkpoint_data['model_state_dict'])

            print(f"Checkpoint cargado exitosamente: {self.best_checkpoint_path}")
            print(f"  Fitness: {checkpoint_data['fitness']:.2f}%, Gen: {checkpoint_data['generation']}, ID: {genome['id']}")
            print(f"  Architecture: {self._format_architecture(genome)}")

            return genome, model
        except Exception as e:
            print(f"Error cargando checkpoint: {e}")
            return None, None

    def _save_fold_progress(self):
        """Publish one consistent progress snapshot from the scheduling thread."""
        snapshot = {'generation': self.generation, 'individuals': self.fold_progress}
        if getattr(self, 'fold_scheduler_progress', None) is not None:
            snapshot['scheduler'] = self.fold_scheduler_progress
        temporary_path = f'{self.fold_progress_path}.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as progress_file:
            json.dump(snapshot, progress_file, indent=2)
        os.replace(temporary_path, self.fold_progress_path)

    def _iter_population_results(self):
        """Use one bounded fold queue for the entire population."""
        num_folds = int(self.config.get('num_folds', 5))
        worker_limit = max(1, int(self.config.get('fold_parallel_workers', 1)))
        adaptive = self.device.type == 'cuda' and self.config.get('fold_auto_workers', True)
        scheduler = VramFoldScheduler(self.config, self.device) if adaptive else None
        if scheduler is not None:
            worker_limit = scheduler.maximum
        self.fold_scheduler_progress = None
        retries = {}
        draining = False
        fold_config = get_data_phase_config(self.config, 'evolution')
        print(
            f"      Fold training: batch_size={fold_config['batch_size']}, "
            f"max_epochs={fold_config['num_epochs']}, "
            f"patience={fold_config['epoch_patience']}, "
            f"AMP={fold_config.get('use_amp', True)} "
            f"({fold_config.get('amp_dtype', 'float16')}), "
            f"cache={fold_config.get('fold_cache_mode', 'none')}"
        )
        train_fold = train_fold_on_cuda_stream if self.device.type == 'cuda' else train_fold_in_thread
        pending = deque()
        cached = []
        keys = {}
        states = {}
        self.fold_progress = {}

        for index, genome in enumerate(self.population):
            key = f"{index + 1}:{genome['id']}"
            keys[index] = key
            skip = bool(genome.pop('skip_next_evaluation', False)) and genome.get('metrics') is not None
            self.fold_progress[key] = {
                'id': genome['id'],
                'status': 'cached' if skip else 'queued',
                'folds': {str(fold): {'status': 'pending', 'fitness': None}
                          for fold in range(1, num_folds + 1)},
                'average_fitness': float(genome['fitness']) if skip else None,
            }
            if skip:
                source = genome.pop('cached_from_generation', self.generation - 1)
                cached.append((index, float(genome['fitness']), None, genome['metrics'], None,
                               'cached_elite', source))
            else:
                genome.pop('cached_from_generation', None)
                # Cost metadata sums actual fold runtimes, excluding time
                # spent waiting in the shared queue.
                states[index] = {'results': {}, 'best_fold': None, 'best_score': None,
                                 'best_model': None, 'elapsed': 0.0}
        # Round-robin order lets every individual enter the queue without
        # reserving a worker for that individual until all five folds finish.
        for fold in range(1, num_folds + 1):
            for index in states:
                pending.append((index, fold))
        self._save_fold_progress()

        futures = {}
        progress_events = SimpleQueue()

        def timed_train_fold(index, fold):
            started = time.perf_counter()
            progress_events.put((index, fold, 'started', None, None))
            worker_config = with_fold_progress_callback(fold_config,
                lambda epoch, total: progress_events.put((index, fold, 'epoch', epoch, total)))
            try:
                result = train_fold(self.population[index], fold, worker_config, self.device)
            except RuntimeError as error:
                if not is_cuda_memory_error(error):
                    raise
                result = FoldMemoryFailure(str(error))
            return result, time.perf_counter() - started

        def collect_progress_events():
            changed = False
            while True:
                try:
                    index, fold, kind, epoch, total = progress_events.get_nowait()
                except Empty:
                    break
                entry = self.fold_progress[keys[index]]['folds'][str(fold)]
                if kind == 'started':
                    entry['started_at'] = time.time()
                else:
                    entry['epoch'] = epoch
                    entry['total_epochs'] = total
                entry['last_update_at'] = time.time()
                changed = True
            if changed:
                self._save_fold_progress()

        def dispatch_folds(executor):
            nonlocal draining
            if scheduler is not None:
                if draining and not futures:
                    # Synchronizing cache cleanup is only needed after a memory
                    # failure, not every time the queue briefly becomes empty.
                    gc.collect()
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                    draining = False
                warming = sum(not self.fold_progress[keys[index]]['folds'][str(fold)].get('epoch')
                              for index, fold in futures.values())
                scheduler.update(len(futures), warming=warming)
                self.fold_scheduler_progress = scheduler.snapshot()
                self.fold_scheduler_progress['draining'] = draining
                if draining:
                    # Admission waits for running folds, but telemetry and the
                    # cooldown must keep advancing while those folds train.
                    self.fold_scheduler_progress['block_reason'] = 'draining_after_memory_error'
                    return
            while pending and len(futures) < worker_limit:
                if scheduler is not None and not scheduler.admit(len(futures)):
                    break
                index, fold = pending.popleft()
                self.fold_progress[keys[index]]['status'] = 'running'
                self.fold_progress[keys[index]]['folds'][str(fold)]['status'] = 'running'
                future = executor.submit(timed_train_fold, index, fold)
                futures[future] = (index, fold)
                if scheduler is not None:
                    scheduler.reserve_slot()
                    self.fold_scheduler_progress = scheduler.snapshot()

        with ThreadPoolExecutor(max_workers=worker_limit) as executor:
            dispatch_folds(executor)
            self._save_fold_progress()
            print(
                f"      Fold queue: {len(futures)} active, {len(pending)} pending. "
                f"Progress: {self.fold_progress_path}"
            )
            for item in cached:
                yield item

            last_heartbeat = time.monotonic()
            while futures or pending:
                completed, _ = wait(tuple(futures), timeout=5, return_when=FIRST_COMPLETED)
                collect_progress_events()
                if not completed and time.monotonic() - last_heartbeat >= 30:
                    completed_count = sum(
                        fold['status'] in {'completed', 'failed'}
                        for entry in self.fold_progress.values() for fold in entry['folds'].values()
                    )
                    active_progress = []
                    for index, fold in futures.values():
                        progress = self.fold_progress[keys[index]]['folds'][str(fold)]
                        epoch = progress.get('epoch')
                        detail = f"epoch {epoch}/{progress['total_epochs']}" if epoch is not None else 'loading'
                        active_progress.append(f"{self.population[index]['id']} fold {fold}: {detail}")
                    print(
                        f"      Fold queue alive: {len(futures)} active, {len(pending)} pending, "
                        f"{completed_count} completed in this generation."
                        + (f" Limit={scheduler.limit}/{scheduler.maximum}; "
                           f"VRAM free={scheduler.free / (1024 ** 3):.2f} GiB; "
                           f"state={'draining_after_memory_error' if draining else scheduler.block_reason}"
                           if scheduler is not None and scheduler.free is not None else '')
                        + (f" Active: {'; '.join(active_progress)}" if active_progress else '')
                    )
                    last_heartbeat = time.monotonic()
                for future in completed:
                    index, fold = futures.pop(future)
                    try:
                        fold_result, fold_elapsed = future.result()
                    except Exception:
                        entry = self.fold_progress[keys[index]]
                        entry['status'] = 'failed'
                        entry['folds'][str(fold)]['status'] = 'failed'
                        self._save_fold_progress()
                        raise
                    if isinstance(fold_result, FoldMemoryFailure):
                        entry = self.fold_progress[keys[index]]
                        fold_entry = entry['folds'][str(fold)]
                        attempt = retries.get((index, fold), 0) + 1
                        retries[index, fold] = attempt
                        fold_entry.update({'status': 'retrying', 'memory_error': fold_result.message,
                                           'retries': attempt, 'fitness': None})
                        if scheduler is not None:
                            scheduler.backoff()
                            self.fold_scheduler_progress = scheduler.snapshot()
                            draining = True
                        if scheduler is not None and attempt <= int(self.config.get('fold_oom_max_retries', 2)):
                            states[index]['elapsed'] += fold_elapsed
                            pending.appendleft((index, fold))
                            print(f"      Retrying individual {self.population[index]['id']} fold {fold} "
                                  f"after CUDA memory failure (retry {attempt}); draining active folds.")
                            self._save_fold_progress()
                            continue
                        print(
                            f"      CUDA memory failure for {self.population[index]['id']} fold {fold} "
                            f"after {attempt} attempts; assigning 0% and continuing. "
                            f"Last error: {fold_result.message}"
                        )
                        fold_result = (fold, 0.0, None, None)
                    result_fold, score, fold_model, metrics = fold_result
                    del fold_result
                    if result_fold != fold:
                        raise ValueError(f"Expected fold {fold}, received {result_fold}")
                    state = states[index]
                    state['elapsed'] += fold_elapsed
                    score = float(score)
                    is_best = (state['best_score'] is None or score > state['best_score'] or
                               (score == state['best_score'] and fold < state['best_fold']))
                    if is_best:
                        # Keep only the best fold's model, on CPU, so finished
                        # folds do not accumulate GPU models behind the queue.
                        if fold_model is not None and self.device.type == 'cuda':
                            fold_model = fold_model.to('cpu')
                        state['best_model'] = fold_model
                        state['best_fold'] = fold
                        state['best_score'] = score
                    state['results'][fold] = (score, None, metrics)
                    del fold_model
                    fold_entry = self.fold_progress[keys[index]]['folds'][str(fold)]
                    fold_entry.update({'status': 'completed' if metrics is not None else 'failed',
                                       'fitness': score, 'duration_seconds': fold_elapsed})
                    print(f"      Individual {self.population[index]['id']} fold {fold}: {score:.2f}%")

                    if len(state['results']) == num_folds:
                        best_fold = state['best_fold']
                        best_score, _, best_metrics = state['results'][best_fold]
                        state['results'][best_fold] = (best_score, state['best_model'], best_metrics)
                        fitness, best_model, aggregated = aggregate_fold_results(
                            self.population[index], self.config, state['results']
                        )
                        elapsed = state['elapsed']
                        fold_failed = any(result[2] is None for result in state['results'].values())
                        self.fold_progress[keys[index]]['status'] = (
                            'completed_with_failed_folds' if fold_failed else 'completed'
                        )
                        self.fold_progress[keys[index]]['average_fitness'] = fitness
                        states.pop(index)
                        state['results'].clear()
                        self._save_fold_progress()
                        yield index, fitness, best_model, aggregated, elapsed, 'trained', None
                        del best_model
                    self._save_fold_progress()
                # Release futures before cleanup/admission; successful futures
                # retain their result models even after the local is deleted.
                if completed:
                    del future
                completed.clear()
                dispatch_folds(executor)
                self._save_fold_progress()

    def evaluate_population(self):
        """Evaluates all individuals in the population and tracks statistics."""
        print(f"\nEvaluating population (Generation {self.generation})...")
        print(f"Processing {len(self.population)} individuals...")

        fitness_scores = []
        all_individual_metrics = [None] * len(self.population)
        best_fitness_so_far = 0.0
        current_global_best_fitness = self.best_individual['fitness'] if self.best_individual else 0.0
        generation_log_lines = []

        for (i, fitness, model, metrics, elapsed_seconds,
             evaluation_status, cached_from_generation) in self._iter_population_results():
            genome = self.population[i]
            print(
                f"   Evaluating individual {i+1}/{len(self.population)} (ID: {genome['id']})"
            )
            print(
                f"      Architecture: {self._format_architecture(genome)}, "
                f"optimizer={genome['optimizer']}, learning_rate={genome['learning_rate']}"
            )

            if evaluation_status == 'cached_elite':
                print(
                    f"      Skipping training: elite selected in generation {cached_from_generation} "
                    f"keeps cached fitness {fitness:.2f}%"
                )
            else:
                genome['fitness'] = fitness

            self._record_cost_metadata(genome, metrics, elapsed_seconds)
            self._sync_selection_metadata_to_metrics(genome, metrics)
            genome['metrics'] = metrics
            fitness_scores.append(fitness)

            individual_summary = {
                'id': genome['id'],
                'fitness': fitness,
                'architecture': self._format_architecture(genome),
                'optimizer': genome['optimizer'],
                'lr': genome['learning_rate'],
                'evaluation_time_seconds': genome.get('evaluation_time_seconds'),
                'parameter_count': genome.get('parameter_count'),
                'metrics': metrics,
                'evaluation_status': evaluation_status
            }
            all_individual_metrics[i] = individual_summary

            if fitness > best_fitness_so_far:
                best_fitness_so_far = fitness

            if fitness > current_global_best_fitness:
                current_global_best_fitness = fitness
                if model is not None:
                    self.save_best_checkpoint(genome, model)
            del model

            # Publish completed individuals immediately for tail -f; the full
            # generation summary is appended after the population completes.
            self._append_generation_log(
                f"Individual {i+1}/{len(self.population)} | ID={genome['id']} | "
                f"status={evaluation_status} | fitness={fitness:.2f}% | "
                f"time={self._format_seconds(genome.get('evaluation_time_seconds'))}s | "
                f"params={self._format_parameter_count(genome.get('parameter_count'))} | "
                f"best_gen={best_fitness_so_far:.2f}% | global_best={current_global_best_fitness:.2f}%"
            )

        self._assign_selection_metadata()
        for genome, individual_summary in zip(self.population, all_individual_metrics):
            metrics = individual_summary['metrics']
            self._sync_selection_metadata_to_metrics(genome, metrics)
            individual_summary['evaluation_time_seconds'] = genome.get('evaluation_time_seconds')
            individual_summary['parameter_count'] = genome.get('parameter_count')
            individual_summary['pareto_rank'] = genome.get('pareto_rank')
            individual_summary['crowding_distance'] = genome.get('crowding_distance')
            individual_summary['selection_objectives'] = genome.get('selection_objectives')

        if fitness_scores:
            avg_fitness = np.mean(fitness_scores)
            max_fitness = np.max(fitness_scores)
            min_fitness = np.min(fitness_scores)
            std_fitness = np.std(fitness_scores)
        else:
            avg_fitness = max_fitness = min_fitness = std_fitness = 0.0

        stats = {
            'generation': self.generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'min_fitness': min_fitness,
            'std_fitness': std_fitness,
            'selection_strategy': self.config.get('selection_strategy', 'fitness'),
            'pareto_objectives': self.config.get('pareto_objectives', DEFAULT_PARETO_OBJECTIVES),
            'individual_metrics': all_individual_metrics
        }
        self.generation_stats.append(stats)
        self.fitness_history.append(max_fitness)

        best_genome = max(self.population, key=lambda x: x['fitness'])
        if self.best_individual is None or best_genome['fitness'] > self.best_individual['fitness']:
            self.best_individual = copy.deepcopy(best_genome)

        if self.best_individual is not None:
            self.best_fitness_overall = max(self.best_fitness_overall, self.best_individual['fitness'])

        if self._uses_pareto_selection():
            sorted_individuals = sorted(
                all_individual_metrics,
                key=lambda individual: pareto_selection_key(individual, self.config),
            )
        else:
            sorted_individuals = sorted(all_individual_metrics, key=lambda x: x['fitness'], reverse=True)

        generation_log_lines.append("=" * 100)
        generation_log_lines.append(f"GENERATION {self.generation} - DETAILED METRICS SUMMARY")
        generation_log_lines.append("=" * 100)
        generation_log_lines.append(
            f"{'ID':<15} {'Arch':<24} {'Rank':<6} {'Crowd':<8} {'Time(s)':<9} "
            f"{'Params':<12} {'Acc':<10} {'Sen':<10} {'Spe':<10} {'Pre':<10} {'F1':<10} {'AUC':<10}"
        )
        generation_log_lines.append("-" * 100)

        for ind in sorted_individuals:
            m = ind['metrics']
            generation_log_lines.append(
                f"{ind['id'][:13]:<15} {ind['architecture']:<24} "
                f"{str(ind.get('pareto_rank', 'NA')):<6} "
                f"{self._format_crowding_distance(ind.get('crowding_distance')):<8} "
                f"{self._format_seconds(ind.get('evaluation_time_seconds')):<9} "
                f"{self._format_parameter_count(ind.get('parameter_count')):<12} "
                f"{m['accuracy']:>6.2f}%   {m['sensitivity']:>6.2f}%   "
                f"{m['specificity']:>6.2f}%   {m['precision']:>6.2f}%   "
                f"{m['f1_score']:>6.2f}%   {m['auc']:>6.2f}%"
            )

        generation_log_lines.append("-" * 100)

        valid_metrics = [ind['metrics'] for ind in all_individual_metrics if ind['metrics']['n_valid_folds'] > 0]
        if valid_metrics:
            gen_avg_acc = np.mean([m['accuracy'] for m in valid_metrics])
            gen_avg_sen = np.mean([m['sensitivity'] for m in valid_metrics])
            gen_avg_spe = np.mean([m['specificity'] for m in valid_metrics])
            gen_avg_pre = np.mean([m['precision'] for m in valid_metrics])
            gen_avg_f1 = np.mean([m['f1_score'] for m in valid_metrics])
            gen_avg_auc = np.mean([m['auc'] for m in valid_metrics])

            generation_log_lines.append(
                f"{'GENERATION AVG':<15} {'':<15} "
                f"{gen_avg_acc:>6.2f}%   {gen_avg_sen:>6.2f}%   "
                f"{gen_avg_spe:>6.2f}%   {gen_avg_pre:>6.2f}%   "
                f"{gen_avg_f1:>6.2f}%   {gen_avg_auc:>6.2f}%"
            )

        generation_log_lines.append("-" * 100)
        generation_log_lines.append(f"GENERATION {self.generation} STATISTICS:")
        generation_log_lines.append(f"   Maximum fitness: {max_fitness:.2f}%")
        generation_log_lines.append(f"   Average fitness: {avg_fitness:.2f}%")
        generation_log_lines.append(f"   Minimum fitness: {min_fitness:.2f}%")
        generation_log_lines.append(f"   Standard deviation: {std_fitness:.2f}%")
        generation_log_lines.append(f"   Best individual: {best_genome['id']} with {best_genome['fitness']:.2f}%")
        generation_log_lines.append(f"   Global best individual: {self.best_individual['id']} with {self.best_individual['fitness']:.2f}%")
        generation_log_lines.append(f"   Selection strategy: {self.config.get('selection_strategy', 'fitness')}")

        if best_genome.get('metrics'):
            bm = best_genome['metrics']
            generation_log_lines.append(f"   Best Individual Metrics (ID: {best_genome['id']}):")
            generation_log_lines.append(f"      Accuracy:     {bm['accuracy']:.2f}% +/- {bm['accuracy_std']:.2f}%")
            generation_log_lines.append(f"      Sensitivity:  {bm['sensitivity']:.2f}% +/- {bm['sensitivity_std']:.2f}%")
            generation_log_lines.append(f"      Specificity:  {bm['specificity']:.2f}% +/- {bm['specificity_std']:.2f}%")
            generation_log_lines.append(f"      Precision:    {bm['precision']:.2f}% +/- {bm['precision_std']:.2f}%")
            generation_log_lines.append(f"      F1-Score:     {bm['f1_score']:.2f}% +/- {bm['f1_score_std']:.2f}%")
            generation_log_lines.append(f"      AUC:          {bm['auc']:.2f}% +/- {bm['auc_std']:.2f}%")
            generation_log_lines.append(
                f"      Cost:         time={self._format_seconds(best_genome.get('evaluation_time_seconds'))}s, "
                f"params={self._format_parameter_count(best_genome.get('parameter_count'))}, "
                f"pareto_rank={best_genome.get('pareto_rank', 'NA')}, "
                f"crowding={self._format_crowding_distance(best_genome.get('crowding_distance'))}"
            )

        generation_log_lines.append("=" * 100)

        self._append_generation_log("\n".join(generation_log_lines))

        print(
            f"Generation {self.generation} summary -> "
            f"max: {max_fitness:.2f}% | avg: {avg_fitness:.2f}% | min: {min_fitness:.2f}% | std: {std_fitness:.2f}%"
        )
        print(f"Detailed generation report appended to: {self.generation_log_path}")

    def selection_and_reproduction(self):
        """Preserve quality, breed within species and introduce fresh candidates."""
        operator_config = get_operator_config(self.config, self.generation, self.current_mutation_rate)
        self._assign_selection_metadata()
        self.population.sort(key=lambda individual: pareto_selection_key(individual, self.config))
        if self.config['speciation_enabled']:
            self._speciate_population()
        size = int(self.config['population_size'])
        # Global elites always include the best scalar-fitness candidate, even
        # when a Pareto extreme would otherwise displace it.
        elite_size = min(size - 1, max(1, int(size * self.config['elite_percentage'])))
        best = max(self.population, key=lambda individual: individual['fitness'])
        selected = [best] + [individual for individual in self.population if individual['id'] != best['id']][:elite_size - 1]
        species_budget = min(size - len(selected) - 1, int(size * self.config['species_elite_fraction'])) if self.config['speciation_enabled'] else 0
        selected_ids = {individual['id'] for individual in selected}
        for specie in self.species.values():
            if species_budget <= 0:
                break
            ranked = sorted(specie['members'], key=lambda individual: pareto_selection_key(individual, self.config))
            for individual in ranked[:int(self.config['species_elite_min'])]:
                if individual['id'] not in selected_ids and species_budget > 0:
                    selected.append(individual)
                    selected_ids.add(individual['id'])
                    species_budget -= 1
        new_population = []
        for individual in selected:
            elite = copy.deepcopy(individual)
            elite['skip_next_evaluation'] = True
            elite['cached_from_generation'] = self.generation - 1
            new_population.append(elite)
        immigrant_count = min(size - len(new_population) - 1, int(size * self.config['random_immigrant_fraction']))
        for _ in range(max(0, immigrant_count)):
            new_population.append(create_random_genome(operator_config))
        templates = get_configured_templates(self.config)
        reference_count = min(size - len(new_population) - 1, int(size * self.config['reference_injection_fraction'])) if templates else 0
        for _ in range(max(0, reference_count)):
            new_population.append(self._create_template_seed_individual(operator_config, len(new_population), 0, self.current_max_conv_layers, self.current_max_fc_layers))
        # One explicit growth probe, separate from immigrants and full models.
        if len(new_population) < size:
            new_population.append(self._create_double_cap_individual())
        while len(new_population) < size:
            if self.config['speciation_enabled'] and self.species:
                species = list(self.species.values())
                # Equal species opportunity prevents a large lineage monopolizing reproduction.
                members = random.choice(species)['members']
                survivors = sorted(members, key=lambda individual: pareto_selection_key(individual, self.config))[:max(1, int(np.ceil(len(members) * self.config['species_survival_rate'])))]
            else:
                survivors = self.population
            parents = [select_tournament(survivors, self.config, self.config['tournament_size']) for _ in range(2)]
            for child in crossover_genomes(*parents, operator_config):
                if len(new_population) < size:
                    new_population.append(mutate_genome(child, operator_config))
        self.population = new_population
        print(f"Generation {self.generation}: {len(selected)} cached elites, {immigrant_count} random immigrants, {reference_count} complete/template injections")

    def tournament_selection(self, tournament_size=None) -> dict:
        return select_tournament(self.population, self.config, tournament_size or self.config['tournament_size'])

    def _update_adaptive_mutation(self):
        """Updates mutation rate based on population diversity."""
        if not self.generation_stats:
            self.current_mutation_rate = self.config['base_mutation_rate']
            return
        
        from ..models.genome_validator import stable_genome_signature
        signatures = {stable_genome_signature(genome, self.config) for genome in self.population}
        diversity_factor = len(signatures) / max(1, len(self.population))
        inverted = 1 - diversity_factor
        new_rate = self.config['base_mutation_rate'] + inverted * self.config['adaptive_mutation_strength']
        stagnation = self.generations_without_improvement / max(1, self.max_generations_without_improvement)
        new_rate += min(1.0, stagnation) * self.config['mutation_stagnation_boost']
        new_rate = max(self.config['mutation_rate_min'], min(self.config['mutation_rate_max'], new_rate))
        self.current_mutation_rate = round(new_rate, 4)
        
        print(f"Adaptive mutation rate: {self.current_mutation_rate}; unique genomes: {len(signatures)}/{len(self.population)}")

    def _update_convergence_state(self):
        """Count each completed evaluation exactly once, including on resume."""
        if self.last_evaluated_generation == self.generation:
            return
        current_best = self.best_individual['fitness'] if self.best_individual else 0.0
        if current_best - self.best_fitness_for_early_stopping >= self.min_improvement_threshold:
            self.best_fitness_for_early_stopping = current_best
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
        self.last_evaluated_generation = self.generation

    def check_convergence(self) -> bool:
        """Read stopping conditions; polling never changes counters."""
        return bool(
            (self.best_individual and self.best_individual['fitness'] >= self.config['fitness_threshold'])
            or self.generation >= self.config['max_generations']
            or (self.last_evaluated_generation == self.generation and self.generation + 1 >= self.config['max_generations'])
            or self.generations_without_improvement >= self.max_generations_without_improvement
        )

    def _print_final_metrics_summary(self):
        """Imprime un resumen detallado de las métricas del mejor individuo encontrado."""
        print(f"\n{'#'*100}")
        print(f"{'#'*35} FINAL BEST INDIVIDUAL SUMMARY {'#'*34}")
        print(f"{'#'*100}")
        
        best = self.best_individual
        print(f"\n🏆 BEST INDIVIDUAL DETAILS:")
        print(f"   ID: {best['id']}")
        print(f"   Architecture: {self._format_architecture(best)}")
        if best.get('architecture_template_id'):
            print(f"   Template: {best.get('architecture_template_id')}")
            print(f"   Template Family: {best.get('architecture_template_family', 'N/A')}")
            print(f"   Template Origin: {best.get('architecture_template_origin', 'N/A')}")
        print(f"   Residual Enabled: {best.get('residual_enabled', False)}")
        if best.get('residual_enabled', False):
            print(f"   Residual Block Size: {best.get('residual_block_size', 2)}")
            print(f"   Residual Projection: {best.get('residual_projection', 'auto')}")
        print(f"   Optimizer: {best['optimizer']}")
        print(f"   Learning Rate: {best['learning_rate']}")
        print(f"   Fitness: {best['fitness']:.2f}%")
        if best.get('evaluation_time_seconds') is not None or best.get('parameter_count') is not None:
            print(f"   Evaluation Time: {self._format_seconds(best.get('evaluation_time_seconds'))}s")
            print(f"   Parameter Count: {self._format_parameter_count(best.get('parameter_count'))}")
        if best.get('pareto_rank') is not None:
            print(f"   Pareto Rank: {best.get('pareto_rank')}")
            print(f"   Crowding Distance: {self._format_crowding_distance(best.get('crowding_distance'))}")
        
        if best.get('metrics'):
            m = best['metrics']
            print(f"\n📊 PERFORMANCE METRICS (5-fold validation fitness):")
            print(f"   {'─'*60}")
            print(f"   {'Metric':<15} {'Mean':<15} {'Std Dev':<15}")
            print(f"   {'─'*60}")
            print(f"   {'Accuracy':<15} {m['accuracy']:>10.2f}%     ± {m['accuracy_std']:>6.2f}%")
            print(f"   {'Sensitivity':<15} {m['sensitivity']:>10.2f}%     ± {m['sensitivity_std']:>6.2f}%")
            print(f"   {'Specificity':<15} {m['specificity']:>10.2f}%     ± {m['specificity_std']:>6.2f}%")
            print(f"   {'Precision':<15} {m['precision']:>10.2f}%     ± {m['precision_std']:>6.2f}%")
            print(f"   {'F1-Score':<15} {m['f1_score']:>10.2f}%     ± {m['f1_score_std']:>6.2f}%")
            print(f"   {'AUC':<15} {m['auc']:>10.2f}%     ± {m['auc_std']:>6.2f}%")
            print(f"   {'─'*60}")
            
            # Formato para tabla científica
            print(f"\n📋 FORMAT FOR SCIENTIFIC TABLES:")
            print(f"   ┌─────────────┬────────────────────────┐")
            print(f"   │ Metric      │ Value (mean ± std)     │")
            print(f"   ├─────────────┼────────────────────────┤")
            print(f"   │ Accuracy    │ {m['accuracy']/100:.2f} ± {m['accuracy_std']/100:.2f}          │")
            print(f"   │ Sensitivity │ {m['sensitivity']/100:.2f} ± {m['sensitivity_std']/100:.2f}          │")
            print(f"   │ Specificity │ {m['specificity']/100:.2f} ± {m['specificity_std']/100:.2f}          │")
            print(f"   │ Precision   │ {m['precision']/100:.2f} ± {m['precision_std']/100:.2f}          │")
            print(f"   │ F1-Score    │ {m['f1_score']/100:.2f} ± {m['f1_score_std']/100:.2f}          │")
            print(f"   │ AUC         │ {m['auc']/100:.2f} ± {m['auc_std']/100:.2f}          │")
            print(f"   └─────────────┴────────────────────────┘")
            
            # Formato LaTeX
            arch = self._format_architecture(best)
            print(f"\n📄 LaTeX FORMAT:")
            latex = f"   Neuroevolution-{arch} & {m['accuracy']/100:.2f} (±{m['accuracy_std']/100:.2f}) & {m['sensitivity']/100:.2f} (±{m['sensitivity_std']/100:.2f}) & {m['specificity']/100:.2f} (±{m['specificity_std']/100:.2f}) & {m['f1_score']/100:.2f} (±{m['f1_score_std']/100:.2f}) & {m['auc']/100:.2f} (±{m['auc_std']/100:.2f}) \\\\\\\\"
            print(latex)
            
            # Formato Markdown
            print(f"\n📝 Markdown FORMAT:")
            markdown = f"   | Neuroevolution-{arch} | {m['accuracy']/100:.2f} (±{m['accuracy_std']/100:.2f}) | {m['sensitivity']/100:.2f} (±{m['sensitivity_std']/100:.2f}) | {m['specificity']/100:.2f} (±{m['specificity_std']/100:.2f}) | {m['f1_score']/100:.2f} (±{m['f1_score_std']/100:.2f}) | {m['auc']/100:.2f} (±{m['auc_std']/100:.2f}) |"
            print(markdown)
            
            # Mostrar métricas por fold si están disponibles
            if m.get('fold_metrics'):
                print(f"\n📈 METRICS BY FOLD:")
                print(f"   {'Fold':<6} {'Acc':<10} {'Sen':<10} {'Spe':<10} {'Pre':<10} {'F1':<10} {'AUC':<10}")
                print(f"   {'-'*66}")
                for fold_num, fold_m in sorted(m['fold_metrics'].items()):
                    if fold_m:
                        print(f"   {fold_num:<6} {fold_m['accuracy']:>6.2f}%   {fold_m['sensitivity']:>6.2f}%   "
                              f"{fold_m['specificity']:>6.2f}%   {fold_m['precision']:>6.2f}%   "
                              f"{fold_m['f1_score']:>6.2f}%   {fold_m['auc']:>6.2f}%")
                print(f"   {'-'*66}")
        else:
            print(f"\n⚠️ No detailed metrics available for best individual")
        
        print(f"\n{'#'*100}")

    def evolve(self) -> dict:
        """Main evolution loop."""
        print("STARTING HYBRID NEUROEVOLUTION PROCESS (adaptive mutation + generation-level early stopping)")
        print("="*80)
        print(f"Configuration:")
        print(f"   Population: {self.config['population_size']} individuals")
        print(f"   Maximum generations: {self.config['max_generations']}")
        print(f"   Target fitness: {self.config['fitness_threshold']}%")
        print(f"   Early stopping (generations): {self.config['early_stopping_generations']} without improvement")
        print(f"   Min improvement threshold: {self.config['min_improvement_threshold']}%")
        print(f"   Selection strategy: {self.config.get('selection_strategy', 'fitness')}")
        if self._uses_pareto_selection():
            print(f"   Pareto objectives: {self.config.get('pareto_objectives', DEFAULT_PARETO_OBJECTIVES)}")
            print(f"   Pareto fitness epsilon: {self.config.get('pareto_fitness_epsilon', 0.0)}")
        print(
            "   Residual search: "
            f"enabled_weight={self.config.get('residual_enabled_weight', 0.35)}, "
            f"disabled_weight={self.config.get('residual_disabled_weight', 0.65)}, "
            f"block_sizes={self.config.get('residual_block_size_options', [2, 3])}"
        )
        print(
            "   Architecture templates: "
            f"seed_fraction={self.config.get('architecture_template_seed_fraction', 0.0)}, "
            f"mutation_weight={self.config.get('architecture_template_mutation_weight', 0.0)}, "
            f"ids={self.config.get('architecture_template_ids', [])}"
        )
        print(f"   Device: {self.device}")
        print("="*80)
        
        self.initialize_population()
        if self.last_evaluated_generation == self.generation and not self.check_convergence():
            self._update_adaptive_mutation()
            self.generation += 1
            self._update_incremental_complexity()
            self.selection_and_reproduction()

        while not self.check_convergence():
            print(f"\n{'='*80}")
            print(f"GENERATION {self.generation}")
            print(f"{'='*80}")
            
            self.evaluate_population()
            self._update_convergence_state()
            self._save_evolution_progress()
            
            if self.check_convergence():
                break
            
            self._update_adaptive_mutation()
            self.generation += 1
            self._update_incremental_complexity()
            self.selection_and_reproduction()
            self._save_evolution_progress()
            print(f"\nPreparing for next generation...")
        
        print(f"\n{'='*80}")
        print(f"EVOLUTION COMPLETED!")
        print(f"{'='*80}")
        print(f"Best individual found:")
        print(f"   ID: {self.best_individual['id']}")
        print(f"   Fitness: {self.best_individual['fitness']:.2f}%")
        print(f"   Total generations processed: {len(self.generation_stats)}")
        print(f"   Generations without improvement: {self.generations_without_improvement}/{self.max_generations_without_improvement}")
        
        self._print_final_metrics_summary()
        
        print("="*80)
        return self.best_individual
