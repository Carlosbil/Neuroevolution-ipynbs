# Tasks: Add Cost-Aware Pareto Selection

## 1. Configuration

- [x] 1.1 Add `selection_strategy` with supported values `fitness` and `pareto`.
- [x] 1.2 Add default `pareto_objectives` for fitness, evaluation time, and
  parameter count.
- [x] 1.3 Add validation for objective names and directions.
- [x] 1.4 Add optional `pareto_fitness_epsilon` and validate it is non-negative.
- [x] 1.5 Keep a fitness-only configuration path for backward-compatible runs.

## 2. Objective Metadata

- [x] 2.1 Measure individual-level evaluation time in
  `HybridNeuroevolution.evaluate_population`.
- [x] 2.2 Store `evaluation_time_seconds` on the genome and metrics.
- [x] 2.3 Compute `parameter_count` with `estimate_genome_parameter_count`.
- [x] 2.4 Store `parameter_count` on the genome and metrics.
- [x] 2.5 Decide and document how missing objective values are treated during
  ranking.

## 3. Pareto Helpers

- [x] 3.1 Implement `dominates` for mixed maximize/minimize objectives.
- [x] 3.2 Implement non-dominated sorting.
- [x] 3.3 Implement crowding-distance assignment inside each front.
- [x] 3.4 Implement a reusable Pareto selection key.
- [x] 3.5 Add metadata assignment for `pareto_rank`, `crowding_distance`, and
  `selection_objectives`.

## 4. Evolution Selection Integration

- [x] 4.1 Assign Pareto metadata after generation evaluation when Pareto
  selection is enabled.
- [x] 4.2 Update elitism sorting to use Pareto rank and crowding distance.
- [x] 4.3 Update tournament selection to use Pareto ranking when configured.
- [x] 4.4 Preserve current max-fitness selection when
  `selection_strategy="fitness"`.
- [x] 4.5 Keep global best checkpoint selection based on scalar fitness for this
  first implementation.

## 5. Logs and Artifacts

- [x] 5.1 Add Pareto rank, crowding distance, parameter count, and evaluation
  time to generation log summaries.
- [x] 5.2 Add objective metadata to `evolution_progress.json`.
- [x] 5.3 Ensure cached elites keep objective metadata.
- [x] 5.4 Update final best-individual summary to show cost metadata when
  present.

## 6. Tests

- [x] 6.1 Add tests for dominance with maximize and minimize objectives.
- [x] 6.2 Add tests for expected non-dominated fronts.
- [x] 6.3 Add tests for crowding-distance tie-breaking.
- [x] 6.4 Add tests for Pareto tournament selection.
- [x] 6.5 Add tests that fitness-only strategy preserves old behavior.
- [x] 6.6 Add tests that evaluated genomes receive time and parameter metadata.
- [x] 6.7 Run relevant tests plus existing residual/Inception tests.

## 7. Documentation

- [x] 7.1 Update README/config docs with the Pareto selection mode.
- [x] 7.2 Update `mejoras/YA IMPLEMENTADAS.md` after implementation.
- [x] 7.3 Note that Pareto and fitness-only runs may not be directly comparable.
