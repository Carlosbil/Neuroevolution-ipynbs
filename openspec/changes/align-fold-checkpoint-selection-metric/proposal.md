# Align Fold Checkpoint Selection Metric

## Why

Evolution currently computes fold fitness from F1-score, but each fold keeps its
best model state using validation/evaluation accuracy. This can preserve an
epoch that improves accuracy while hurting F1, especially when class balance or
the sensitivity/specificity tradeoff matters.

For this project, F1 is the evolutionary objective reported as fitness. The
model state saved per fold should therefore be selected by F1 by default, or by
an explicit objective metric when the experiment chooses another target.

## What Changes

- Add a configurable fold model-selection metric.
- Default fold model selection to the fitness objective, currently F1-score.
- Use the selected metric, not hard-coded accuracy, when deciding whether to
  keep `best_state` inside a fold.
- Store and report the selected metric value, selected epoch, and metric name in
  fold metadata.
- Apply the same selection semantics in final 5-fold evaluation.
- Add tests that fail if F1 fitness still uses accuracy-selected fold
  checkpoints.

## Scope

This change covers:

- `neuroevolution/config.py`
- `neuroevolution/evolution/fitness.py`
- `neuroevolution/evaluation/cross_validation.py`
- relevant tests under `tests/`
- README/improvement docs where checkpoint or best-epoch selection is described

It is compatible with the separate validation/test protocol change. If both
changes are implemented, this metric should be computed on the validation split
for model selection and test should remain final-report only.

## Non-goals

- Do not change the Conv1D architecture search space.
- Do not change residual or Inception genome behavior.
- Do not implement multi-objective evolution in this change.
- Do not change the physical fold file format.
- Do not require users to optimize F1 forever; the point is to make the target
  explicit and configurable.

## Success Criteria

- When the configured selection metric is `f1_score`, fold `best_state` is
  selected by F1 and not accuracy.
- The default behavior aligns fold checkpoint selection with the current
  evolutionary fitness objective.
- Supported metric names are validated up front.
- Fold metrics include the selected metric name and best selected metric value.
- Final evaluation uses the same configurable selection metric for choosing the
  best epoch.
- Tests cover at least one case where accuracy and F1 prefer different epochs.
