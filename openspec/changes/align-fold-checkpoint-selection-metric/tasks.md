# Tasks: Align Fold Checkpoint Selection Metric

## 1. Configuration

- [x] 1.1 Add `fitness_metric` with default `f1_score` to
  `neuroevolution/config.py`.
- [x] 1.2 Add `fold_selection_metric` with default `fitness_metric`.
- [x] 1.3 Add validation for supported metric names and the
  `fitness_metric` alias.
- [x] 1.4 Add optional `metric_improvement_threshold` or document that
  `improvement_threshold` applies to the selected metric.

## 2. Metric Helpers

- [x] 2.1 Extract classification metric calculation into a reusable helper.
- [x] 2.2 Add helper logic to resolve `fold_selection_metric` to an actual
  metric name.
- [x] 2.3 Add helper logic to fetch a numeric metric value with safe fallback.
- [x] 2.4 Ensure `recall` aliases to `sensitivity` if exposed as a metric name.

## 3. Evolution Fitness

- [x] 3.1 Update `train_fold_in_thread` to compute full validation metrics at
  each validation interval.
- [x] 3.2 Replace `best_acc` with `best_selection_score` and select `best_state`
  using the resolved selection metric.
- [x] 3.3 Return fold score from the configured `fitness_metric` or keep F1 as
  the explicit default objective.
- [x] 3.4 Add `selection_metric`, `best_selection_score`, and `best_epoch` to
  per-fold metrics.
- [x] 3.5 Update logs so they report the selected metric instead of hard-coded
  accuracy.

## 4. Final Evaluation

- [x] 4.1 Update `evaluate_single_fold` in
  `neuroevolution/evaluation/cross_validation.py` to use the same selection
  metric resolution.
- [x] 4.2 Replace hard-coded `best_acc` naming with neutral selected-metric
  naming.
- [x] 4.3 Preserve compatibility fields only if existing notebooks need them.
- [x] 4.4 Update final evaluation messages to mention the configured selection
  metric.

## 5. Tests

- [x] 5.1 Add unit tests for metric-name validation and alias resolution.
- [x] 5.2 Add tests for metric calculation on simple binary predictions.
- [x] 5.3 Add a regression test where accuracy and F1 prefer different epochs
  and F1 selection chooses the F1-best state.
- [x] 5.4 Add a test that `fold_selection_metric="accuracy"` keeps the old
  accuracy-selection behavior when explicitly requested.
- [x] 5.5 Add or update final evaluation tests for selected-metric metadata.
- [x] 5.6 Run relevant tests plus existing residual/Inception smoke tests.

## 6. Documentation

- [x] 6.1 Update README/config documentation to explain `fitness_metric` and
  `fold_selection_metric`.
- [x] 6.2 Update `mejoras/YA IMPLEMENTADAS.md` after implementation to mark the
  checkpoint metric alignment as implemented.
- [x] 6.3 Note that runs selected by accuracy and runs selected by F1 may not be
  directly comparable.
