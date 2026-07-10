# Design

## Current Problem

`neuroevolution/evolution/fitness.py` currently loads `X_train`, `X_val`, and `X_test`, then creates:

```python
x_eval = torch.cat([x_val_tensor, x_test_tensor], dim=0)
y_eval = torch.cat([y_val_tensor, y_test_tensor], dim=0)
test_dataset = torch.utils.data.TensorDataset(x_eval, y_eval)
```

That combined loader is used for:

- periodic epoch validation,
- best epoch state selection,
- fold metrics,
- returned fold score,
- averaged evolutionary fitness,
- final held-out fold evaluation through `evaluation/cross_validation.py`.

The protocol must instead keep validation and test separate.

## Target Split Contract

Introduce a single split-aware loader contract for model-training workflows:

```python
FoldLoaders(
    train: DataLoader,
    validation: DataLoader,
    test: DataLoader,
)
```

The concrete implementation can be a dataclass or a small typed tuple, but callers must refer to the three splits by name to avoid accidental tuple-order mistakes.

## Evolution Fitness Flow

For each genome and each fold:

1. Build the model from the genome.
2. Train using `train` only.
3. At `validation_frequency_epochs`, evaluate on `validation` only.
4. Select the best per-fold state using a configured validation metric.
5. Restore the best validation-selected state.
6. Compute final validation metrics for that fold.
7. Return the validation F1-score as the fold fitness score.

The averaged evolutionary fitness remains:

```text
fitness = mean(validation_f1_score over all folds)
```

This matches the current implementation's actual F1-score behavior while removing test leakage. Documentation that says "fitness = average accuracy" must be corrected.

## Checkpoint Flow

The global best checkpoint should continue to be saved during evolution, but it must be based only on validation-derived fitness.

Checkpoint metadata should make this explicit:

- `selection_split`: `"validation"`
- `fitness_metric`: `"f1_score"`
- `checkpoint_metric`: `"f1_score"`
- `test_evaluated`: `false`

The checkpoint file must not claim to be a test-selected model.

## Final Evaluation Flow

Final evaluation should use the selected genome architecture, not test-selected weights.

For each fold:

1. Instantiate a fresh model from the selected genome.
2. Train on the fold train split.
3. Use the fold validation split for early stopping/checkpoint selection.
4. Restore the best validation-selected state.
5. Evaluate once on the fold test split.
6. Report test metrics.

This gives a clean held-out estimate while still allowing validation-based model selection inside each final fold run.

The existing optional `use_pretrained`/checkpoint initialization in final evaluation should be removed or disabled by default. If kept for exploratory runs, it must be clearly marked as non-final and must not be used for article test metrics unless the weights were produced without touching any test split.

## API Shape

Recommended changes:

- Add `FoldLoaders` in `neuroevolution/evolution/fitness.py` or a shared evaluation/data module.
- Replace `load_fold_data(...) -> (train_loader, test_loader)` with a split-aware function such as:

```python
load_fold_loaders(fold_number, config, device) -> FoldLoaders
```

- Keep a temporary compatibility wrapper only if needed, but update internal code to the explicit split-aware API.
- Rename misleading variables such as `fold_test_loader` when they are actually validation loaders.

## Testing Strategy

Add focused tests with small synthetic arrays:

- Verify loader returns distinct train, validation, and test datasets.
- Verify evolution-time training/evaluation does not iterate over test loader.
- Verify fitness equals a validation-derived metric, not test-derived metric.
- Verify final evaluation computes test metrics only after validation-selected state restoration.
- Verify docs/config text no longer says test is part of training/evolution evaluation.

Mocking or sentinel labels can be used to fail fast if test samples are accessed during evolution.

## Documentation Updates

Update README and notebook markdown to state:

- Each fold contains train/validation/test.
- Evolution fitness uses validation F1-score averaged across folds.
- Checkpoints are selected by validation metric.
- Test is held out for final metrics only.
- The procedure is not "test-monitored cross-validation"; it is a repeated/folded train-validation-test evaluation protocol.
