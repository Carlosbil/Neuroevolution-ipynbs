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
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from ..genetics.genome import create_random_genome
from ..genetics.mutation import mutate_genome
from ..genetics.crossover import crossover_genomes
from ..genetics.innovation import build_innovation_genes, append_structural_event
from ..models.genome_validator import validate_and_fix_genome
from ..models.evolvable_cnn import EvolvableCNN
from .fitness import evaluate_fitness, evaluate_population_concurrent


class HybridNeuroevolution:
    """Main class that implements hybrid neuroevolution with 5-fold CV and adaptive mutation."""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        self.generation_stats = []

        self.run_artifacts_dir = self.config.get('artifacts_dir', 'artifacts/test_audio')
        os.makedirs(self.run_artifacts_dir, exist_ok=True)
        self.best_checkpoint_path = None
        self.progress_json_path = os.path.join(self.run_artifacts_dir, 'evolution_progress.json')
        self.generation_log_path = os.path.join(self.run_artifacts_dir, 'generation_progress.txt')

        # Early stopping configuration at generation level
        self.generations_without_improvement = 0
        self.best_fitness_overall = -float('inf')
        self.best_fitness_for_early_stopping = -float('inf')
        self.min_improvement_threshold = config.get('min_improvement_threshold', 0.1)
        self.max_generations_without_improvement = config.get('early_stopping_generations', 10)

        # Incremental evolution defaults
        self.config.setdefault('complexity_step_generations', 3)
        self.config.setdefault('initial_max_conv_layers', self.config['min_conv_layers'])
        self.config.setdefault('initial_max_fc_layers', self.config['min_fc_layers'])
        self.config.setdefault('incremental_growth_probability', 0.7)

        # Genetic speciation defaults
        self.config.setdefault('speciation_threshold', 0.45)
        self.config.setdefault('species_elite_min', 1)
        self.config.setdefault('species_survival_rate', 0.5)

        # neo_conc concurrent-evaluation defaults
        self.config.setdefault('concurrent_individuals', 5)
        self.config.setdefault('fold_rotation_strategy', 'sequential')
        self.config.setdefault('fold_rotation_offset', 1)
        self.fold_history = []

        self.config['current_max_conv_layers'] = self.config['initial_max_conv_layers']
        self.config['current_max_fc_layers'] = self.config['initial_max_fc_layers']

        self.species = {}

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
            'generation': self.generation,
            'population': self._to_json_serializable(self.population),
            'best_individual': self._to_json_serializable(self.best_individual),
            'fitness_history': self._to_json_serializable(self.fitness_history),
            'generation_stats': self._to_json_serializable(self.generation_stats),
            'generations_without_improvement': self.generations_without_improvement,
            'best_fitness_overall': self.best_fitness_overall,
            'best_fitness_for_early_stopping': self.best_fitness_for_early_stopping,
            'best_checkpoint_path': self.best_checkpoint_path,
            'fold_history': self.fold_history,
        }

        with open(self.progress_json_path, 'w', encoding='utf-8') as progress_file:
            json.dump(progress_data, progress_file, indent=2)

    def _load_evolution_progress(self) -> bool:
        """Loads evolution state from JSON file if exists."""
        if not os.path.exists(self.progress_json_path):
            return False

        try:
            with open(self.progress_json_path, 'r', encoding='utf-8') as progress_file:
                progress_data = json.load(progress_file)

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
            self.fold_history = progress_data.get('fold_history', [])
            return True
        except Exception as e:
            print(f"WARNING: Could not load evolution progress: {e}")
            return False

    def _update_incremental_complexity(self):
        """Updates complexity caps based on current generation."""
        step = max(1, self.config['complexity_step_generations'])
        stage = max(0, self.generation // step)

        self.config['current_max_conv_layers'] = min(
            self.config['max_conv_layers'],
            self.config['initial_max_conv_layers'] + stage
        )
        self.config['current_max_fc_layers'] = min(
            self.config['max_fc_layers'],
            self.config['initial_max_fc_layers'] + stage
        )

    def _enforce_complexity_caps(self, genome: dict) -> dict:
        """Enforces incremental complexity caps on a genome."""
        genome['num_conv_layers'] = min(genome['num_conv_layers'], self.config['current_max_conv_layers'])
        genome['num_fc_layers'] = min(genome['num_fc_layers'], self.config['current_max_fc_layers'])
        genome = validate_and_fix_genome(genome, self.config)
        genome['innovation_genes'] = build_innovation_genes(genome)
        return genome

    def _create_double_cap_individual(self) -> dict:
        """Creates one individual using doubled layer caps for the current generation."""
        base_conv_cap = max(
            self.config['min_conv_layers'],
            self.config.get('current_max_conv_layers', self.config['max_conv_layers'])
        )
        base_fc_cap = max(
            self.config['min_fc_layers'],
            self.config.get('current_max_fc_layers', self.config['max_fc_layers'])
        )

        target_conv_layers = min(self.config['max_conv_layers'], base_conv_cap * 2)
        target_fc_layers = min(self.config['max_fc_layers'], base_fc_cap * 2)

        max_safe_conv_layers = int(np.log2(self.config['sequence_length'] / 4))
        target_conv_layers = max(
            self.config['min_conv_layers'],
            min(target_conv_layers, max_safe_conv_layers)
        )
        target_fc_layers = max(self.config['min_fc_layers'], target_fc_layers)

        boosted_config = copy.deepcopy(self.config)
        boosted_config['current_max_conv_layers'] = target_conv_layers
        boosted_config['current_max_fc_layers'] = target_fc_layers

        genome = create_random_genome(boosted_config)
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

    def compatibility_distance(self, g1: dict, g2: dict) -> float:
        """Combines topology differences and innovation mismatch for speciation."""
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

        return 0.45 * topo + 0.45 * innovation_mismatch + 0.10 * numeric

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
            self._update_incremental_complexity()
            print(f"Resuming evolution from generation {self.generation} using: {self.progress_json_path}")
            return

        with open(self.generation_log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("Generation progress log\n")

        self._update_incremental_complexity()
        print(f"Initializing population of {self.config['population_size']} individuals...")
        print(
            f"Incremental caps at generation {self.generation}: "
            f"conv<={self.config['current_max_conv_layers']}, fc<={self.config['current_max_fc_layers']}"
        )

        reserve_double_cap_slot = self.config['population_size'] > 1
        regular_population_size = (
            self.config['population_size'] - 1
            if reserve_double_cap_slot
            else self.config['population_size']
        )

        self.population = []
        for i in range(regular_population_size):
            genome = create_random_genome(self.config)

            # Start from simple networks and grow gradually
            genome['num_conv_layers'] = self.config['min_conv_layers']
            genome['num_fc_layers'] = self.config['min_fc_layers']
            genome = validate_and_fix_genome(genome, self.config)

            if i % 3 == 0 and random.random() < self.config['incremental_growth_probability']:
                if genome['num_conv_layers'] < self.config['current_max_conv_layers']:
                    genome['num_conv_layers'] += 1
                    append_structural_event(genome, 'init_add_conv_layer', {'index_in_population': i})
                elif genome['num_fc_layers'] < self.config['current_max_fc_layers']:
                    genome['num_fc_layers'] += 1
                    append_structural_event(genome, 'init_add_fc_layer', {'index_in_population': i})
                genome = validate_and_fix_genome(genome, self.config)

            genome = self._enforce_complexity_caps(genome)
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

        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            try:
                os.remove(self.best_checkpoint_path)
                print(f"      Checkpoint anterior eliminado: {self.best_checkpoint_path}")
            except Exception as e:
                print(f"      Error eliminando checkpoint anterior: {e}")

        checkpoint_filename = f"best_model_gen{self.generation}_id{genome['id']}_fitness{genome['fitness']:.2f}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'genome': genome,
            'generation': self.generation,
            'fitness': genome['fitness'],
            'config': self.config
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.best_checkpoint_path = checkpoint_path
            self._save_evolution_progress()
            print(f"      Nuevo checkpoint guardado: {checkpoint_path}")
            print(f"        Fitness: {genome['fitness']:.2f}%, ID: {genome['id']}, Gen: {self.generation}")
        except Exception as e:
            print(f"      Error guardando checkpoint: {e}")

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

            return genome, model
        except Exception as e:
            print(f"Error cargando checkpoint: {e}")
            return None, None

    def _select_fold_for_generation(self) -> int:
        """Picks the single fold used to evaluate this generation.

        Sequential rotation (default): gen 0 -> fold 1, gen 1 -> fold 2, ...
        Random rotation: uniformly random fold each generation.
        Each generation uses a DIFFERENT fold so the population does not
        overfit to a single split across the run.
        """
        num_folds = max(1, int(self.config.get('num_folds', 5)))
        offset = int(self.config.get('fold_rotation_offset', 1))
        strategy = str(self.config.get('fold_rotation_strategy', 'sequential')).lower()

        if strategy == 'random':
            fold_num = random.randint(1, num_folds)
        else:
            fold_num = ((self.generation + (offset - 1)) % num_folds) + 1
        return fold_num

    def evaluate_population(self):
        """Evaluates all individuals on a SINGLE rotated fold, concurrently.

        Differences vs the original implementation:
          * One fold per generation (rotated), not 5 folds per individual.
          * Up to `config['concurrent_individuals']` individuals train in
            parallel on the same GPU using torch.cuda.Stream() + AMP.
        """
        fold_num = self._select_fold_for_generation()
        self.fold_history.append({'generation': self.generation, 'fold': fold_num})

        print(f"\nEvaluating population (Generation {self.generation})...")
        print(f"Processing {len(self.population)} individuals on fold {fold_num} "
              f"(rotation strategy: {self.config.get('fold_rotation_strategy', 'sequential')})")

        for i, genome in enumerate(self.population):
            print(
                f"   Queued individual {i+1}/{len(self.population)} (ID: {genome['id']}) "
                f"| {genome['num_conv_layers']} conv + {genome['num_fc_layers']} fc | "
                f"opt={genome['optimizer']} lr={genome['learning_rate']}"
            )

        eval_results = evaluate_population_concurrent(
            self.population, fold_num, self.config, self.device
        )

        fitness_scores = []
        all_individual_metrics = []
        best_fitness_so_far = 0.0
        current_global_best_fitness = self.best_individual['fitness'] if self.best_individual else 0.0
        generation_log_lines = [f"Evaluation fold for generation {self.generation}: {fold_num}"]

        for i, (genome, (fitness, model, metrics)) in enumerate(zip(self.population, eval_results)):
            genome['fitness'] = fitness
            genome['metrics'] = metrics
            genome['evaluation_fold'] = fold_num
            fitness_scores.append(fitness)

            individual_summary = {
                'id': genome['id'],
                'fitness': fitness,
                'architecture': f"{genome['num_conv_layers']}conv+{genome['num_fc_layers']}fc",
                'optimizer': genome['optimizer'],
                'lr': genome['learning_rate'],
                'metrics': metrics,
                'evaluation_fold': fold_num,
            }
            all_individual_metrics.append(individual_summary)

            if fitness > best_fitness_so_far:
                best_fitness_so_far = fitness

            if fitness > current_global_best_fitness:
                current_global_best_fitness = fitness
                if model is not None:
                    self.save_best_checkpoint(genome, model)

            generation_log_lines.append(
                f"Individual {i+1}/{len(self.population)} | ID={genome['id']} | "
                f"fold={fold_num} | fitness={fitness:.2f}% | "
                f"best_gen={best_fitness_so_far:.2f}% | global_best={current_global_best_fitness:.2f}%"
            )

        if fitness_scores:
            avg_fitness = float(np.mean(fitness_scores))
            max_fitness = float(np.max(fitness_scores))
            min_fitness = float(np.min(fitness_scores))
            std_fitness = float(np.std(fitness_scores))
        else:
            avg_fitness = max_fitness = min_fitness = std_fitness = 0.0

        stats = {
            'generation': self.generation,
            'evaluation_fold': fold_num,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'min_fitness': min_fitness,
            'std_fitness': std_fitness,
            'individual_metrics': all_individual_metrics
        }
        self.generation_stats.append(stats)
        self.fitness_history.append(max_fitness)

        best_genome = max(self.population, key=lambda x: x['fitness'])
        if self.best_individual is None or best_genome['fitness'] > self.best_individual['fitness']:
            self.best_individual = copy.deepcopy(best_genome)

        if self.best_individual is not None:
            self.best_fitness_overall = max(self.best_fitness_overall, self.best_individual['fitness'])

        sorted_individuals = sorted(all_individual_metrics, key=lambda x: x['fitness'], reverse=True)

        generation_log_lines.append("=" * 100)
        generation_log_lines.append(
            f"GENERATION {self.generation} (fold {fold_num}) - DETAILED METRICS SUMMARY"
        )
        generation_log_lines.append("=" * 100)
        generation_log_lines.append(
            f"{'ID':<15} {'Arch':<15} {'Acc':<10} {'Sen':<10} {'Spe':<10} {'Pre':<10} {'F1':<10} {'AUC':<10}"
        )
        generation_log_lines.append("-" * 100)

        for ind in sorted_individuals:
            m = ind['metrics']
            generation_log_lines.append(
                f"{ind['id'][:13]:<15} {ind['architecture']:<15} "
                f"{m['accuracy']:>6.2f}%   {m['sensitivity']:>6.2f}%   "
                f"{m['specificity']:>6.2f}%   {m['precision']:>6.2f}%   "
                f"{m['f1_score']:>6.2f}%   {m['auc']:>6.2f}%"
            )

        generation_log_lines.append("-" * 100)

        valid_metrics = [ind['metrics'] for ind in all_individual_metrics if ind['metrics']['n_valid_folds'] > 0]
        if valid_metrics:
            gen_avg_acc = float(np.mean([m['accuracy'] for m in valid_metrics]))
            gen_avg_sen = float(np.mean([m['sensitivity'] for m in valid_metrics]))
            gen_avg_spe = float(np.mean([m['specificity'] for m in valid_metrics]))
            gen_avg_pre = float(np.mean([m['precision'] for m in valid_metrics]))
            gen_avg_f1 = float(np.mean([m['f1_score'] for m in valid_metrics]))
            gen_avg_auc = float(np.mean([m['auc'] for m in valid_metrics]))

            generation_log_lines.append(
                f"{'GENERATION AVG':<15} {'':<15} "
                f"{gen_avg_acc:>6.2f}%   {gen_avg_sen:>6.2f}%   "
                f"{gen_avg_spe:>6.2f}%   {gen_avg_pre:>6.2f}%   "
                f"{gen_avg_f1:>6.2f}%   {gen_avg_auc:>6.2f}%"
            )

        generation_log_lines.append("-" * 100)
        generation_log_lines.append(f"GENERATION {self.generation} STATISTICS (fold {fold_num}):")
        generation_log_lines.append(f"   Maximum fitness: {max_fitness:.2f}%")
        generation_log_lines.append(f"   Average fitness: {avg_fitness:.2f}%")
        generation_log_lines.append(f"   Minimum fitness: {min_fitness:.2f}%")
        generation_log_lines.append(f"   Standard deviation: {std_fitness:.2f}%")
        generation_log_lines.append(f"   Best individual: {best_genome['id']} with {best_genome['fitness']:.2f}%")
        generation_log_lines.append(
            f"   Global best individual: {self.best_individual['id']} with {self.best_individual['fitness']:.2f}%"
        )
        generation_log_lines.append("=" * 100)

        self._append_generation_log("\n".join(generation_log_lines))

        print(
            f"Generation {self.generation} (fold {fold_num}) summary -> "
            f"max: {max_fitness:.2f}% | avg: {avg_fitness:.2f}% | "
            f"min: {min_fitness:.2f}% | std: {std_fitness:.2f}%"
        )
        print(f"Detailed generation report appended to: {self.generation_log_path}")

    def selection_and_reproduction(self):
        """Selects best individuals and creates new generation through crossover and mutation."""
        print(f"\nStarting selection and reproduction...")
        
        # Sort by fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        reserve_double_cap_slot = self.config['population_size'] > 1
        regular_target_size = (
            self.config['population_size'] - 1
            if reserve_double_cap_slot
            else self.config['population_size']
        )

        elite_size = max(1, int(self.config['population_size'] * self.config['elite_percentage']))
        elite_size = min(elite_size, regular_target_size)
        elite = self.population[:elite_size]
        
        print(f"Selecting {elite_size} elite individuals:")
        for i, individual in enumerate(elite):
            print(f"   Elite {i+1}: {individual['id']} (fitness: {individual['fitness']:.2f}%)")
        
        new_population = copy.deepcopy(elite)
        offspring_needed = regular_target_size - len(new_population)
        
        print(f"Creating {offspring_needed} new individuals through crossover and mutation...")
        offspring_created = 0
        
        while len(new_population) < regular_target_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child1, child2 = crossover_genomes(parent1, parent2, self.config)
            child1 = mutate_genome(child1, self.config)
            
            if len(new_population) < regular_target_size:
                new_population.append(child1)
            
            child2 = mutate_genome(child2, self.config)
            if len(new_population) < regular_target_size:
                new_population.append(child2)
            
            offspring_created += 2
            if offspring_created % 4 == 0:
                print(f"   Created {min(offspring_created, offspring_needed)} of {offspring_needed} new individuals...")
        
        self.population = new_population[:regular_target_size]
        if reserve_double_cap_slot:
            self.population.append(self._create_double_cap_individual())
        
        print(f"New generation created with {len(self.population)} individuals")
        print(f"   Elite preserved: {elite_size}")
        print(f"   New individuals: {len(self.population) - elite_size}")

    def tournament_selection(self, tournament_size: int = 3) -> dict:
        """Selects best individual from a random tournament."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x['fitness'])

    def _update_adaptive_mutation(self):
        """Updates mutation rate based on population diversity."""
        if not self.generation_stats:
            self.config['current_mutation_rate'] = self.config['base_mutation_rate']
            return
        
        last_std = self.generation_stats[-1]['std_fitness']
        diversity_factor = min(1.0, last_std / 10.0)
        inverted = 1 - diversity_factor
        new_rate = self.config['base_mutation_rate'] + (inverted - 0.5) * 0.4
        new_rate = max(self.config['mutation_rate_min'], min(self.config['mutation_rate_max'], new_rate))
        self.config['current_mutation_rate'] = round(new_rate, 4)
        
        print(f"Adaptive mutation rate updated to {self.config['current_mutation_rate']} (std_fitness={last_std:.2f})")

    def check_convergence(self) -> bool:
        """
        Verifica criterios de convergencia:
        1. Target fitness alcanzado
        2. Máximo de generaciones alcanzado
        3. Early stopping: sin mejora en N generaciones
        4. Estancamiento detectado en últimas generaciones
        """
        # Criterion 1: Target fitness reached
        if self.best_individual and self.best_individual['fitness'] >= self.config['fitness_threshold']:
            print(f"\n✅ Target fitness reached! ({self.best_individual['fitness']:.2f}% >= {self.config['fitness_threshold']}%)")
            return True
        
        # Criterion 2: Maximum generations reached
        if self.generation >= self.config['max_generations']:
            print(f"\n⏱️ Maximum generations reached ({self.generation}/{self.config['max_generations']})")
            return True
        
        # Criterion 3: Early stopping - no improvement in N generations
        if self.generation > 0:
            current_best = self.best_individual['fitness'] if self.best_individual else 0.0
            improvement = current_best - self.best_fitness_for_early_stopping
            
            if improvement >= self.min_improvement_threshold:
                self.best_fitness_for_early_stopping = current_best
                self.generations_without_improvement = 0
                print(f"\n🔄 Improvement detected: {improvement:.2f}% | Generations without improvement: {self.generations_without_improvement}")
            else:
                self.generations_without_improvement += 1
                print(f"\n⏳ No significant improvement | Generations without improvement: {self.generations_without_improvement}/{self.max_generations_without_improvement}")
                
                if self.generations_without_improvement >= self.max_generations_without_improvement:
                    print(f"\n🛑 EARLY STOPPING: No improvement for {self.max_generations_without_improvement} generations")
                    print(f"   Best fitness plateau: {self.best_fitness_overall:.2f}%")
                    return True
        
        # Criterion 4: Stagnation in last 3 generations
        if len(self.fitness_history) >= 3:
            recent = self.fitness_history[-3:]
            if max(recent) - min(recent) < 0.5:
                print(f"\n📉 Stagnation detected in last 3 generations (all within {max(recent) - min(recent):.2f}%)")
        
        return False

    def _print_final_metrics_summary(self):
        """Imprime un resumen detallado de las métricas del mejor individuo encontrado."""
        print(f"\n{'#'*100}")
        print(f"{'#'*35} FINAL BEST INDIVIDUAL SUMMARY {'#'*34}")
        print(f"{'#'*100}")
        
        best = self.best_individual
        print(f"\n🏆 BEST INDIVIDUAL DETAILS:")
        print(f"   ID: {best['id']}")
        print(f"   Architecture: {best['num_conv_layers']} Conv1D layers + {best['num_fc_layers']} FC layers")
        print(f"   Optimizer: {best['optimizer']}")
        print(f"   Learning Rate: {best['learning_rate']}")
        print(f"   Fitness: {best['fitness']:.2f}%")
        
        if best.get('metrics'):
            m = best['metrics']
            print(f"\n📊 PERFORMANCE METRICS (5-Fold Cross-Validation):")
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
            arch = f"{best['num_conv_layers']}Conv1D+{best['num_fc_layers']}FC"
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
        print(f"   Device: {self.device}")
        print("="*80)
        
        self.initialize_population()
        
        while not self.check_convergence():
            print(f"\n{'='*80}")
            print(f"GENERATION {self.generation}")
            print(f"{'='*80}")
            
            self.evaluate_population()
            self._save_evolution_progress()
            
            if self.check_convergence():
                break
            
            self._update_adaptive_mutation()
            self.generation += 1
            self._update_incremental_complexity()
            self.selection_and_reproduction()
            print(f"\nPreparing for next generation...")
        
        print(f"\n{'='*80}")
        print(f"EVOLUTION COMPLETED!")
        print(f"{'='*80}")
        print(f"Best individual found:")
        print(f"   ID: {self.best_individual['id']}")
        print(f"   Fitness: {self.best_individual['fitness']:.2f}%")
        print(f"   Origin generation: {self.generation}")
        print(f"   Total generations processed: {self.generation + 1}")
        print(f"   Generations without improvement: {self.generations_without_improvement}/{self.max_generations_without_improvement}")
        
        self._print_final_metrics_summary()
        
        print("="*80)
        return self.best_individual
