# Tasks: Separate Validation and Test During Evolution

## 1. Fold Loader Semantics

- [x] 1.1 Add explicit split selection to the fold DataLoader helper in
  `neuroevolution/evolution/fitness.py`.
- [x] 1.2 Include the requested evaluation split in the DataLoader cache key.
- [x] 1.3 Ensure `eval_split="validation"` returns train + validation loaders
  without loading or concatenating test data for fitness.
- [x] 1.4 Support explicit test loading for final evaluation without changing the
  `.npy` fold file layout.
- [x] 1.5 Keep or isolate any `validation_and_test` compatibility mode so it is
  never used by evolutionary fitness.

## 2. Evolution Fitness

- [x] 2.1 Update `train_fold_in_thread` to use validation-only loaders during
  evolution.
- [x] 2.2 Rename local variables, logs, and comments from test/eval wording to
  validation wording where they refer to evolutionary fitness.
- [x] 2.3 Add split metadata to returned metrics, for example
  `evaluation_split="validation"` and `fitness_split="validation"`.
- [x] 2.4 Verify epoch best-state selection and fold fitness are computed from
  validation data only.
- [x] 2.5 Ensure checkpoint saving in `HybridNeuroevolution` still works with the
  best validation-selected fold model.

## 3. Final Evaluation

- [x] 3.1 Update `neuroevolution/evaluation/cross_validation.py` to load train,
  validation, and test separately.
- [x] 3.2 Use validation metrics for epoch selection in final evaluation.
- [x] 3.3 Use test metrics only for final reported fold metrics.
- [x] 3.4 Update final evaluation print messages that currently describe
  `val+test` or "same methodology as evolution".
- [x] 3.5 Preserve the existing aggregate output fields where possible so
  notebooks and report utilities keep working.

## 4. Tests

- [x] 4.1 Add tests for split-aware fold loading with tiny temporary `.npy`
  folds.
- [x] 4.2 Add a regression test proving evolution evaluation loaders contain
  validation samples only, not validation plus test.
- [x] 4.3 Add a test proving cache keys differ for validation and test loaders.
- [x] 4.4 Add or update final evaluation tests to confirm test metrics are
  reported from the test split after validation-selected checkpoint restore.
- [x] 4.5 Run the relevant test subset plus existing residual/Inception tests.

## 5. Documentation

- [x] 5.1 Update README evaluation wording to state that evolution uses
  validation-only fitness and final evaluation reserves test.
- [x] 5.2 Update `mejoras/YA IMPLEMENTADAS.md` after implementation to mark the
  two requested improvements as implemented.
- [x] 5.3 Add a short migration note that old runs using `val+test` fitness are
  not directly comparable to the cleaned protocol.
