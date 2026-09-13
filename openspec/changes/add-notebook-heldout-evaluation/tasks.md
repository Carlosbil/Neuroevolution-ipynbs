# Tasks

## 1. Final Evaluation Persistence

- [x] Update `evaluate_5fold_cross_validation` to save its timestamped JSON result beneath `config['artifacts_dir']`.
- [x] Add validation-selection and test-evaluation provenance to the returned and serialized result.
- [x] Return the persisted result path and ensure JSON serialization handles non-native numeric values.

## 2. Notebook Execution Flow

- [x] Import `evaluate_5fold_cross_validation` in `test.ipynb`.
- [x] Add a dedicated held-out evaluation cell directly after evolution completes.
- [x] Save the return value as `final_test_results` and report the persisted artifact path.
- [x] Make the cell fail clearly when final evaluation produces no results.

## 3. Notebook Reporting

- [x] Add a test-results table with per-fold, mean, and standard-deviation rows for accuracy, sensitivity, specificity, F1, and AUC.
- [x] Render a labelled confusion matrix for every successful test fold.
- [x] Label all new output as held-out test evaluation and keep validation fitness separate.

## 4. Verification

- [x] Add focused tests for artifact-directory JSON persistence and result provenance.
- [x] Add focused tests for table/confusion-matrix presentation using mocked results.
- [ ] Run the relevant test suite and validate the notebook JSON structure without running the full experiment.
