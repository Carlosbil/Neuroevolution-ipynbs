# Design: Cost-Aware Pareto Selection

## Current State

The evolution engine evaluates each genome and stores:

- `fitness`
- aggregated metrics
- generation summaries

Selection is scalar:

- generation elites are selected from `population.sort(key=lambda x:
  x["fitness"], reverse=True)`;
- `tournament_selection` returns the sampled genome with max `fitness`;
- global best and convergence also use scalar `fitness`.

The project already has a deterministic `estimate_genome_parameter_count`
helper in `neuroevolution/models/genome_validator.py`, but its value is not
stored as an objective. Evaluation time is not recorded.

## Target Behavior

Keep `fitness` as the primary performance metric, but add a selection layer that
can rank individuals by multiple objectives:

```text
maximize: fitness or configured primary metric
minimize: evaluation_time_seconds
minimize: parameter_count
```

The default selection strategy for this change should be configurable:

```python
"selection_strategy": "pareto",  # or "fitness"
"pareto_objectives": [
    {"name": "fitness", "direction": "maximize"},
    {"name": "evaluation_time_seconds", "direction": "minimize"},
    {"name": "parameter_count", "direction": "minimize"},
],
"pareto_tie_breaker": "crowding_distance",
"pareto_fitness_epsilon": 0.0,
```

`selection_strategy="fitness"` should preserve current behavior for direct
comparisons with old runs.

## Objective Collection

### Evaluation time

Wrap each individual evaluation in `HybridNeuroevolution.evaluate_population`
with `time.perf_counter()`:

```python
start = time.perf_counter()
fitness, model, metrics = evaluate_fitness(...)
elapsed = time.perf_counter() - start
```

Store:

```python
genome["evaluation_time_seconds"] = elapsed
metrics["evaluation_time_seconds"] = elapsed
```

Optionally record per-fold timing inside `train_fold_in_thread` so logs can
show where time is spent, but individual-level timing is enough for Pareto
selection.

### Parameter count

After genome validation, compute:

```python
parameter_count = estimate_genome_parameter_count(genome, config)
```

Store:

```python
genome["parameter_count"] = parameter_count
metrics["parameter_count"] = parameter_count
```

If estimation fails, store `None` and treat the missing value as worst possible
for minimization during Pareto ranking.

## Pareto Ranking

Add selection helpers, preferably in `neuroevolution/genetics/selection.py`:

- `dominates(a, b, objectives, epsilon=0.0)`
- `non_dominated_sort(population, objectives)`
- `assign_pareto_metadata(population, objectives)`
- `pareto_selection_key(individual, config)`

Dominance rule:

- `a` dominates `b` if `a` is no worse than `b` on every objective and strictly
  better on at least one objective.
- For maximize objectives, higher is better.
- For minimize objectives, lower is better.
- `pareto_fitness_epsilon` can treat tiny fitness differences as equivalent so
  very small F1 gains do not dominate large cost savings.

Metadata:

```python
genome["pareto_rank"] = 0  # first front
genome["crowding_distance"] = ...
genome["selection_objectives"] = {
    "fitness": ...,
    "evaluation_time_seconds": ...,
    "parameter_count": ...,
}
```

Lower `pareto_rank` is better. Higher `crowding_distance` is better inside a
front because it preserves spread across the tradeoff surface.

## Elitism and Parent Selection

When `selection_strategy="pareto"`:

- assign Pareto metadata after evaluating the generation;
- select elites by `(pareto_rank ascending, crowding_distance descending,
  fitness descending)`;
- tournament selection should choose the best sampled individual by the same
  key;
- generation logs should show rank, time, parameters, and fitness.

When `selection_strategy="fitness"`:

- keep existing sorting and tournament behavior.

## Global Best and Checkpointing

Keep global best checkpoint selection based on scalar `fitness` for this first
implementation. That keeps the scientific "best F1 model" output stable and
prevents Pareto tradeoff policy from silently changing final reported results.

The Pareto metadata should still be saved in progress JSON and generation logs,
so a user can identify cheaper first-front candidates after the run.

A later change can add a separate "deployment candidate" checkpoint selected by
Pareto rank.

## Logging and Artifacts

Add columns or fields to generation summaries:

- `pareto_rank`
- `crowding_distance`
- `evaluation_time_seconds`
- `parameter_count`
- objective values used for selection

`evolution_progress.json` should keep these fields on each genome and in
`individual_metrics`.

## Interactions With Other Proposed Changes

This change complements:

- validation/test separation: Pareto selection should use validation-derived
  fitness, while test remains final-only.
- configurable fold selection metric: if `fitness_metric` changes, Pareto's
  primary objective can still use scalar `fitness`.
- multi-fidelity evaluation: later phases can use Pareto ranking to choose
  which candidates are promoted.

## Testing Strategy

Use small dictionaries rather than real models for most Pareto tests.

Required tests:

- dominance for maximize/minimize objectives;
- non-dominated sorting creates expected fronts;
- crowding distance gives boundary points high priority;
- Pareto tournament prefers a non-dominated cheaper candidate over a dominated
  expensive candidate;
- `selection_strategy="fitness"` preserves max-fitness selection;
- evaluated genomes receive `evaluation_time_seconds` and `parameter_count`
  metadata without breaking existing metric summaries.

## Risks and Mitigations

- **Risk:** Evaluation time is noisy.
  **Mitigation:** use time as a secondary objective and keep fitness as primary
  checkpoint metric; optional epsilon reduces overreaction to tiny differences.
- **Risk:** A cheap but weak model survives too often.
  **Mitigation:** Pareto rank requires tradeoff quality; users can keep
  `selection_strategy="fitness"` or tune objectives.
- **Risk:** Metadata churn in progress JSON grows.
  **Mitigation:** store compact scalar values, not raw timing traces unless
  explicitly enabled.
