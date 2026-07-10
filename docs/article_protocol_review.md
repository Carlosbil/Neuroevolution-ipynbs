# Article Protocol Review

## Scope

This note records the local evidence reviewed for the train/validation/test protocol, synthetic data usage, and validation terminology.

## Findings

- The package loader exposes separate `train`, `validation`, and `test` loaders through `FoldLoaders`.
- Evolution-time fitness uses validation metrics only; test is reserved for final evaluation.
- `data/sets/folds_5/files_real_N` contains 180 train, 60 validation, and 60 test samples per fold, with balanced labels in each split. This supports a 60/20/20 sample-level protocol.
- The local real audio directories contain 100 unique subject IDs inferred from filenames.
- No subject-ID manifest was found in `data/sets/folds_5`, `data/sets.zip`, or tracked project files.
- Because fold `.npy` files do not preserve subject IDs, strict subject-level isolation and full test-subject coverage across folds cannot be audited from the current artifacts.

## Synthetic Data Assessment

The current synthetic fold folders should not be used as article-grade final evaluation evidence without an accompanying subject manifest:

- `files_real_40_1e5_N` has approximately 7140-7200 train samples, 2400 validation samples, and 60 test samples per fold. The validation size is not compatible with the real-only validation split, so validation cannot be assumed real-only.
- `files_all_real_syn_n` explicitly contains mixed real+synthetic train, validation, and test data according to notebook documentation and sample counts.

For article-grade claims that synthetic data is used only in training, generate or provide fold artifacts where:

- train may contain real plus synthetic samples from train subjects only,
- validation contains real samples only,
- test contains real samples only,
- every split includes a subject-ID manifest.

## Recommended Article Wording

Use:

> We evaluated the selected architecture using five stratified train/validation/test partitions. Within each partition, training data were used for weight updates, validation data for early stopping and model selection, and the held-out test subset only for final metrics.

Avoid calling the protocol "classical k-fold cross-validation" unless a manifest proves that every subject appears in exactly one held-out test fold across the full evaluation.

## Checkpoint Metric

The implementation should be described consistently as validation-F1 selected:

> The best epoch state and global checkpoint were selected by maximizing validation F1-score, the same metric used as evolutionary fitness.

Accuracy is reported as an evaluation metric, but it is not the primary checkpoint-selection criterion in the current configuration.
