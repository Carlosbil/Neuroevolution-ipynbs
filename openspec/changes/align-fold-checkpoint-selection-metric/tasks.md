# Tasks: Align Fold Checkpoint Selection Metric

## 1. Configuration

- [ ] 1.1 Add `fitness_metric` with default `f1_score` to
  `neuroevolution/config.py`.
- [ ] 1.2 Add `fold_selection_metric` with default `fitness_metric`.
- [ ] 1.3 Add validation for supported metric names and the
  `fitness_metric` alias.
- [ ] 1.4 Add optional `metric_improvement_threshold` or document that
  `improvement_threshold` applies to the selected metric.

## 2. Metric Helpers

- [ ] 2.1 Extract classification metric calculation into a reusable helper.
- [ ] 2.2 Add helper logic to resolve `fold_selection_metric` to an actual
  metric name.
- [ ] 2.3 Add helper logic to fetch a numeric metric value with safe fallback.
- [ ] 2.4 Ensure `recall` aliases to `sensitivity` if exposed as a metric name.

## 3. Evolution Fitness

- [ ] 3.1 Update `train_fold_in_thread` to compute full validation metrics at
  each validation interval.
- [ ] 3.2 Replace `best_acc` with `best_selection_score` and select `best_state`
  using the resolved selection metric.
- [ ] 3.3 Return fold score from the configured `fitness_metric` or keep F1 as
  the explicit default objective.
- [ ] 3.4 Add `selection_metric`, `best_selection_score`, and `best_epoch` to
  per-fold metrics.
- [ ] 3.5 Update logs so they report the selected metric instead of hard-coded
  accuracy.

## 4. Final Evaluation

- [ ] 4.1 Update `evaluate_single_fold` in
  `neuroevolution/evaluation/cross_validation.py` to use the same selection
  metric resolution.
- [ ] 4.2 Replace hard-coded `best_acc` naming with neutral selected-metric
  naming.
- [ ] 4.3 Preserve compatibility fields only if existing notebooks need them.
- [ ] 4.4 Update final evaluation messages to mention the configured selection
  metric.

## 5. Tests

- [ ] 5.1 Add unit tests for metric-name validation and alias resolution.
- [ ] 5.2 Add tests for metric calculation on simple binary predictions.
- [ ] 5.3 Add a regression test where accuracy and F1 prefer different epochs
  and F1 selection chooses the F1-best state.
- [ ] 5.4 Add a test that `fold_selection_metric="accuracy"` keeps the old
  accuracy-selection behavior when explicitly requested.
- [ ] 5.5 Add or update final evaluation tests for selected-metric metadata.
- [ ] 5.6 Run relevant tests plus existing residual/Inception smoke tests.

## 6. Documentation

- [ ] 6.1 Update README/config documentation to explain `fitness_metric` and
  `fold_selection_metric`.
- [ ] 6.2 Update `mejoras/YA IMPLEMENTADAS.md` after implementation to mark the
  checkpoint metric alignment as implemented.
- [ ] 6.3 Note that runs selected by accuracy and runs selected by F1 may not be
  directly comparable.
