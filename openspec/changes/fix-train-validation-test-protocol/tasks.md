# Tasks

## 1. Data Loading Contract

- [x] Add a split-aware `FoldLoaders` contract with named `train`, `validation`, and `test` loaders.
- [x] Refactor fold loading so validation and test arrays are never concatenated.
- [x] Update cache keys and loader cache behavior so split-aware loaders remain deterministic.
- [x] Rename misleading variables such as `fold_test_loader` where they represent validation/evaluation loaders.

## 2. Evolution-Time Fitness

- [x] Update `train_fold_in_thread` to train only on `train`.
- [x] Evaluate epoch progress only on `validation`.
- [x] Select the best per-fold state only from validation metrics.
- [x] Return validation F1-score as the fold score used for evolutionary fitness.
- [x] Ensure aggregated fitness metrics are validation metrics, not test metrics.

## 3. Global Checkpoint Selection

- [x] Save global best checkpoints only when validation-derived fitness improves.
- [x] Add checkpoint metadata documenting `selection_split="validation"`, `fitness_metric="f1_score"`, and `checkpoint_metric="f1_score"`.
- [x] Ensure no test metrics are required or computed before checkpoint save.

## 4. Final Evaluation

- [x] Refactor `evaluate_single_fold` to receive train, validation, and test loaders separately.
- [x] Train the selected genome architecture from scratch per fold by default.
- [x] Use validation only for early stopping and best-state selection.
- [x] Evaluate the restored best state once on test.
- [x] Remove or clearly mark pretrained checkpoint initialization as non-final/exploratory.
- [x] Ensure final reported metrics are test metrics only.

## 5. Tests

- [x] Add unit tests for split-aware loader construction.
- [x] Add a regression test proving evolution fitness does not access test data.
- [x] Add a regression test proving checkpoint selection does not access test data.
- [x] Add a final-evaluation test proving test metrics are computed only after validation selection.
- [x] Run the relevant test suite and document any skipped heavy/integration tests.

## 6. Documentation And Article Checklist

- [x] Update README protocol description.
- [x] Update notebook markdown cells that say fitness is average accuracy.
- [x] Update notebook markdown cells that describe final evaluation with checkpoints.
- [x] Mark checklist section 1 items according to the corrected implementation.
- [x] Add a short protocol note suitable for the article methods section.

## Protocol Summary

- [x] `train` is used for model weight updates.
- [x] `validation` is used after epochs during training/evolution for monitoring, early stopping, checkpoint selection, evolutionary fitness, and winner selection.
- [x] `test` is used only after the full algorithm finishes, for final held-out metrics.

## 7. Synthetic Data And Subject-Level Splits

- [x] Review contradictions about synthetic data usage across train/validation/test.
- [x] Confirm synthetic data is used only for training.
- [x] Confirm validation and test contain only real data.
- [x] Verify folds are partitioned strictly at subject level.
- [x] Check whether a manifest exists with subject IDs per fold.
- [x] Verify no synthetic sample derived from validation/test subjects appears in training.
- [x] Align Table 5, Section 4.3, and methodological wording if inconsistencies are found.

Review outcome:

- Default configuration now uses `files_real_N` (`dataset_id="real_N"`) for article-safe real-only train/validation/test folds.
- Synthetic fold variants are documented as exploratory unless a subject-ID manifest proves validation/test remain real-only and synthetic samples are derived only from training subjects.
- No subject-ID manifest was found in local fold artifacts or `data/sets.zip`; strict subject-level claims remain unauditable from current `.npy` files.

## 8. Experimental Validation Terminology

- [x] Review whether the described protocol is classical cross-validation.
- [x] Check whether it is five subject-stratified hold-out partitions.
- [x] Confirm the 60/20/20 train/validation/test scheme.
- [x] Verify whether test subsets cover the 100 subjects across folds.
- [x] Adjust methodological terminology if it is not classical cross-validation.

Review outcome:

- Wording now avoids classical cross-validation claims in README, package messages, and notebook-facing text.
- `files_real_N` confirms 180/60/60 samples per fold, i.e. 60/20/20 by samples.
- The local real audio folders contain 100 inferred subject IDs, but test-subject coverage across folds cannot be proven without a fold manifest.

## 9. Checkpoint Metric Review

- [x] Review the contradiction between article Sections 3.4 and 4.4.
- [x] Confirm whether checkpoints are selected by F1-score or accuracy.
- [x] Align the checkpoint metric with the evolutionary fitness metric.
- [x] Evaluate whether F1-score or another balanced metric should be the primary selection criterion.

Review outcome:

- Evolutionary fitness is the mean validation `f1_score` across folds.
- Checkpoint selection now has explicit defaults: `fitness_metric="f1_score"` and `checkpoint_metric="f1_score"`.
- Accuracy remains a reported descriptive metric, not the primary selection criterion.
- Article text should state that checkpoints/best states are selected by validation F1-score, matching the evolutionary fitness metric.
