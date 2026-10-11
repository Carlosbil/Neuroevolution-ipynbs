# Design

## Current State

`evaluate_5fold_cross_validation(best_genome, config, device, ...)` already:

1. creates fresh models for five folds;
2. trains with each fold's `train` data;
3. selects the best epoch through `validation` metrics; and
4. evaluates the restored state on `test`.

It returns a dictionary with `fold_results`, aggregate mean/std metrics, and metadata. Each fold result includes its confusion matrix. The function currently saves `5fold_cv_results_<timestamp>.json` in the process working directory and only prints article-oriented output.

`test.ipynb` invokes `neuroevolution.evolve()` but does not import or call that evaluator, and contains no final-test results cell.

## Target Flow

```text
evolve() -> best_genome (validation-selected)
         -> evaluate_5fold_cross_validation(...)
              -> train / validation state selection / test evaluation per fold
              -> results JSON in CONFIG['artifacts_dir']
              -> returned held-out results
         -> notebook tables and confusion-matrix figures
```

The evaluation call must be in a separate, clearly labelled notebook cell immediately after evolution. This makes an intentional second compute phase visible to notebook users and prevents the validation-derived `best_genome['fitness']` from being presented as a test result.

## Result Persistence

The evaluator shall resolve the output directory from `config['artifacts_dir']`, create it if necessary, and write a timestamped JSON result there. It shall include an explicit `selection_split: "validation"` and `evaluation_split: "test"` in the serialized results, so the file remains auditable outside the notebook. Values originating from NumPy or tensors must be converted to JSON-compatible built-in values.

The evaluator should return the saved file path in the result dictionary (for example, `results_path`) so the notebook can report a clickable/reproducible location without reconstructing a filename.

## Notebook Presentation

Add imports for `evaluate_5fold_cross_validation` and the plotting/tabulation dependencies already used by the project. The evaluation cell shall:

- execute only after a valid `best_genome` exists;
- call the evaluator with `best_genome`, `CONFIG`, `device`, and the same neuroevolution instance for context only;
- retain the returned object as `final_test_results`;
- fail clearly if no fold succeeds; and
- state that the results are final held-out test metrics.

Follow it with a presentation cell that builds a table with rows for each fold and `Mean`/`Std` rows. The columns shall be accuracy, sensitivity, specificity, F1, and AUC. It shall render or print the table in a format usable in the article workflow, and plot one labelled confusion matrix for every successful fold using the matrices returned by final evaluation.

No validation metrics from `best_genome` may be mixed into the final-test table.

## Testing Strategy

Use small mocked fold results and a temporary artifacts directory to verify that:

- persisted files are created inside the configured artifacts directory;
- serialized output contains split provenance and the returned path;
- all fold and aggregate metrics are present in the table input; and
- one confusion matrix is generated per successful fold without requiring a GPU or data files.

Keep existing tests for the train/validation/test protocol unchanged; they establish the correctness of the evaluation logic this notebook now exposes.
