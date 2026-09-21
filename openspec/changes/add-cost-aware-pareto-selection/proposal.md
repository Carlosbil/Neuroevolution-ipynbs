# Add Cost-Aware Pareto Selection

## Why

The current evolutionary loop selects elites and parents by a single scalar:
fitness. In practice, fitness is the average F1-score across folds. This can
push the search toward architectures that are slightly better but much slower
or much larger, which is especially costly in a single-GPU workflow.

The project already estimates parameter counts for genomes and records rich
metrics, but selection does not yet use model cost. Adding Pareto-based
multiobjective selection lets the algorithm prefer candidates that are good on
F1 while also being reasonable in evaluation time and parameter count.

## What Changes

- Measure and store evaluation time per individual and per fold.
- Estimate and store model parameter count per evaluated genome.
- Add configurable multiobjective selection with Pareto ranking.
- Use objectives such as:
  - maximize F1/fitness;
  - minimize evaluation time;
  - minimize parameter count.
- Update elite preservation and tournament selection to prefer lower Pareto
  rank and maintain diversity with a tie-breaker such as crowding distance.
- Keep scalar `fitness` for compatibility, reporting, convergence, and existing
  notebooks.
- Add logs and progress JSON fields that expose Pareto rank, objective values,
  evaluation time, and parameter count.
- Add tests for dominance, non-dominated sorting, cost-aware tournament
  selection, and backward-compatible scalar fitness selection.

## Scope

This change covers:

- `neuroevolution/evolution/engine.py`
- `neuroevolution/evolution/fitness.py`
- `neuroevolution/genetics/selection.py`
- `neuroevolution/models/genome_validator.py` integration for parameter count
- relevant tests under `tests/`
- README or improvement docs that describe selection behavior

## Non-goals

- Do not replace F1 as the primary scientific metric.
- Do not remove scalar `fitness`.
- Do not implement multi-fidelity evaluation in this change.
- Do not change residual or Inception architecture construction.
- Do not require GPU telemetry; elapsed wall-clock time is enough for the first
  implementation.
- Do not optimize final reported test metrics by time or parameter count unless
  a later change explicitly defines that reporting policy.

## Success Criteria

- Each evaluated individual records F1/fitness, evaluation time, and estimated
  parameter count.
- Pareto rank is computed from configured objectives.
- Elitism and tournament selection can use Pareto rank rather than only scalar
  fitness.
- A high-F1 but very expensive model can be dominated or deprioritized by a
  similarly strong cheaper model.
- Existing fitness-only behavior remains available through config.
- Progress logs and JSON make cost-aware selection decisions auditable.
