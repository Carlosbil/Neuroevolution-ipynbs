# Add Notebook Held-Out Evaluation

## Summary

Connect `test.ipynb` to the existing `evaluate_5fold_cross_validation` final-evaluation API immediately after evolution completes. The notebook will run the selected architecture once per train/validation/test fold, persist the held-out test results under the configured artifacts directory, and present the per-fold and aggregate results in an article-ready table.

## Why

The package already implements a final evaluation that retrains the selected architecture, selects its state with validation F1, and evaluates that state on held-out test data. `test.ipynb` currently stops after evolutionary selection, so its displayed metrics are validation-selection metrics rather than independent final test results. In addition, the evaluator writes its JSON result to the current working directory instead of the run artifacts directory.

## Goals

- Invoke held-out five-fold evaluation from `test.ipynb` after `best_genome = neuroevolution.evolve()`.
- Preserve a clear boundary between evolutionary validation metrics and final test metrics.
- Store a JSON result in `CONFIG['artifacts_dir']`.
- Display test metrics and each fold's confusion matrix in the notebook.
- Build a reusable table containing per-fold values plus mean and standard deviation for accuracy, sensitivity, specificity, F1, and AUC.
- Add focused tests for artifact persistence and result presentation helpers.

## Non-Goals

- Changing fold generation, subject manifests, synthetic-data policy, baselines, or ablations.
- Changing the validation-based evolutionary fitness or checkpoint protocol.
- Executing the full, computationally expensive experiment as part of this change.
- Adding confidence intervals or statistical tests; those are a later work item.

## Impact

Primary paths:

- `test.ipynb`
- `neuroevolution/evaluation/cross_validation.py`
- a small result-presentation helper, if needed, under `neuroevolution/evaluation/`
- focused tests for the changed behavior

The notebook's final report will explicitly identify all reported metrics as held-out test results and will save them alongside the evolution artifacts.
