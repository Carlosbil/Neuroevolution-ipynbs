# Design: Align Fold Checkpoint Selection Metric

## Current State

`neuroevolution/evolution/fitness.py` computes evolutionary fitness as the
average fold F1-score. However, inside `train_fold_in_thread`, the best epoch is
selected by accuracy:

```python
best_acc = 0.0
...
if acc > (best_acc + improvement_threshold):
    best_acc = acc
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
```

`neuroevolution/evaluation/cross_validation.py` follows the same pattern with
`best_acc`, and its logs state that the best model is saved by validation
accuracy.

This means the model returned for checkpointing can be inconsistent with the
fitness metric used by evolution.

## Proposed Configuration

Add explicit metric configuration:

```python
"fitness_metric": "f1_score",
"fold_selection_metric": "fitness_metric",
"metric_improvement_threshold": None,
```

Semantics:

- `fitness_metric` controls the fold metric aggregated into evolutionary
  fitness. It should default to the current behavior: `f1_score`.
- `fold_selection_metric="fitness_metric"` means "use whatever
  `fitness_metric` is configured to use".
- Users may set `fold_selection_metric` directly to one supported metric.
- `metric_improvement_threshold=None` should fall back to the existing
  `improvement_threshold` for backward-compatible tuning.

Supported metric names should include:

- `accuracy`
- `precision`
- `sensitivity`
- `recall` as an alias of `sensitivity`
- `specificity`
- `f1_score`
- `auc`

If a metric cannot be computed for an epoch, it should resolve to `0.0` rather
than crashing the whole run, matching current AUC behavior.

## Metric Helpers

Create small helpers in `fitness.py` or a shared evaluation module:

```python
compute_classification_metrics(y_true, y_pred, y_prob) -> dict
resolve_metric_name(config, key="fold_selection_metric") -> str
metric_value(metrics, metric_name) -> float
```

The helper should compute the same fields currently returned per fold:

- accuracy
- sensitivity
- specificity
- precision
- f1_score
- auc

The training loop can use the helper at validation time instead of computing
only `correct/total`. This keeps epoch selection and final metrics consistent.

## Evolution Fold Selection

In `train_fold_in_thread`:

1. Resolve `selection_metric`.
2. During periodic validation/evaluation, collect predictions and probabilities.
3. Compute the metric dictionary.
4. Compare `metrics[selection_metric]` against `best_selection_score`.
5. Store `best_state`, `best_epoch`, `best_selection_score`, and optionally
   `best_selection_metrics`.
6. After training, reload `best_state`.
7. Return final fold metrics and metadata:

```python
metrics["selection_metric"] = selection_metric
metrics["best_selection_score"] = best_selection_score
metrics["best_epoch"] = best_epoch
```

The fold score returned to `evaluate_fitness` should come from
`fitness_metric`, not hard-coded F1, if the implementation adds configurable
fitness. If configurable fitness is considered too broad for the first pass,
keep fitness as F1 and set `selection_metric` default to `f1_score`.

## Final Evaluation

Update `cross_validation.py` to use the same metric resolution for best-epoch
selection. Its logs should say:

```text
Saving best model based on validation f1_score
```

or the configured metric name. The result dictionary should replace or
supplement `best_train_acc` with neutral names:

```python
"selection_metric": selection_metric,
"best_selection_score": best_selection_score,
"best_epoch": best_epoch,
```

Keeping `best_train_acc` as a compatibility alias is acceptable if notebooks
still read it, but new code should use the neutral names.

## Interaction With Validation/Test Separation

This change is about *which metric* selects the best epoch. The split used for
selection is controlled by the validation/test separation change.

If both changes are present:

- selection metric is computed on validation;
- final reported metrics are computed on test;
- fitness is aggregated from validation according to `fitness_metric`.

If this change is implemented first, it should still improve consistency by
selecting from the current evaluation loader with F1 rather than accuracy.

## Testing Strategy

Add tests with a fake or tiny validation loop where two candidate epochs have
different preferences:

- epoch A has higher accuracy;
- epoch B has higher F1.

With `fold_selection_metric="f1_score"`, the selected best state must be epoch
B. With `fold_selection_metric="accuracy"`, it must be epoch A.

Also test:

- config validation rejects unknown metric names;
- `fold_selection_metric="fitness_metric"` resolves to `fitness_metric`;
- final evaluation result metadata includes the selected metric name and best
  score.

## Risks and Mitigations

- **Risk:** Computing F1/AUC every validation pass is slightly more expensive
  than accuracy.
  **Mitigation:** validation already runs periodically; the cost is small
  compared with training Conv1D models.
- **Risk:** Existing logs/notebooks expect `best_train_acc`.
  **Mitigation:** keep compatibility fields during the transition and add
  clearer neutral fields.
- **Risk:** Changing the default selected epoch may alter benchmark numbers.
  **Mitigation:** document that the new default aligns selection with the
  reported objective and is scientifically preferable.
