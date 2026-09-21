# Design: Separate Validation and Test During Evolution

## Current State

`neuroevolution/evolution/fitness.py` has a single `load_fold_data` helper that
returns `(fold_train_loader, fold_test_loader)`. Internally it loads train,
validation, and test arrays, then creates:

```python
x_eval = torch.cat([x_val_tensor, x_test_tensor], dim=0)
y_eval = torch.cat([y_val_tensor, y_test_tensor], dim=0)
test_dataset = TensorDataset(x_eval, y_eval)
```

`train_fold_in_thread` uses that combined loader for epoch validation,
early-stopping checkpoint selection, and final fold metrics. As a result, the
evolutionary fitness signal includes test examples.

`neuroevolution/evaluation/cross_validation.py` delegates to the same loader and
documents the final fold loader as `val+test combined`, so the same conflation
appears in the final evaluation path.

## Target Protocol

The protocol should become:

```text
Evolution:
  train split      -> gradient updates
  validation split -> epoch validation, best-state selection, fitness metrics
  test split       -> not loaded or not used

Final evaluation:
  train split      -> gradient updates
  validation split -> epoch validation and best-state selection
  test split       -> final reported metrics only
```

This keeps the search objective honest while preserving the existing 5-fold
structure.

## Loader Shape

Introduce a split-aware fold loader in `fitness.py`. The simplest local shape is
to keep the existing public name but add an explicit mode:

```python
load_fold_data(fold_number, config, device, eval_split="validation")
```

Supported values:

- `"validation"`: return `(train_loader, validation_loader)`.
- `"test"`: return `(train_loader, test_loader)`.
- `"validation_and_test"`: optional compatibility path for old callers, but not
  used by evolution.
- `"all"`: return `(train_loader, validation_loader, test_loader)` if the final
  evaluation code benefits from explicit split outputs.

The implementation should avoid concatenating validation and test unless
`eval_split="validation_and_test"` is requested explicitly. New evolutionary
code must request validation-only.

The DataLoader cache key must include the requested evaluation split so cached
validation loaders and test loaders cannot be mixed.

## Evolution Fitness

Rename local variables in `train_fold_in_thread` to reflect split semantics:

- `fold_train_loader`
- `fold_validation_loader`

The training loop should:

1. train on `fold_train_loader`;
2. validate periodically on `fold_validation_loader`;
3. keep the best state based on validation score;
4. compute final fold metrics on validation data;
5. aggregate fold validation F1 as evolutionary fitness.

Metric dictionaries returned by `evaluate_fitness` should either keep the
current keys for compatibility or add clear split metadata such as:

```python
metrics["evaluation_split"] = "validation"
aggregated_metrics["fitness_split"] = "validation"
```

The logs should say "Validation" instead of "Test" for evolution metrics.

## Final Evaluation

Update `cross_validation.py` so final evaluation receives separate validation
and test loaders. `evaluate_single_fold` should:

1. train on the train loader;
2. periodically evaluate validation accuracy or the configured selection metric
   on the validation loader;
3. restore the best validation-selected state;
4. evaluate final metrics on the test loader;
5. return both validation-selection metadata and test metrics.

This preserves the existing final report structure while changing the source of
reported metrics to test-only.

The final report should avoid saying it uses the same methodology as evolution
if that means combined `val+test`. It should instead state that validation
selects the best epoch and test is used only for final reporting.

## Backward Compatibility

The fold files remain unchanged. Existing config keys such as `num_folds`,
`batch_size`, `fold_cache_mode`, and DataLoader worker settings remain valid.

Any compatibility mode that returns combined validation and test data should be
clearly named and documented as not suitable for scientific fitness evaluation.
The default for evolution-facing calls should be validation-only.

## Testing Strategy

Add unit tests that monkeypatch or create tiny `.npy` folds with distinguishable
validation and test labels/sizes. The tests should assert:

- evolution's `load_fold_data(..., eval_split="validation")` returns only the
  validation count in the evaluation loader;
- `evaluate_fitness` or `train_fold_in_thread` requests validation-only loaders;
- final evaluation uses a validation loader for best-state selection and a test
  loader for final metrics;
- the DataLoader cache key differs between validation and test split requests.

Existing residual and Inception tests should remain unchanged.

## Risks and Mitigations

- **Risk:** Fitness values drop because fewer samples are used during evolution.
  **Mitigation:** This is expected and scientifically cleaner. Report the change
  in docs so old and new runs are not compared blindly.
- **Risk:** Existing notebooks rely on variable names such as `fold_test_loader`.
  **Mitigation:** Keep compatibility wrappers where useful, but make new names
  explicit in core modules and final output.
- **Risk:** Cached DataLoaders leak combined split behavior.
  **Mitigation:** include split mode in cache keys and add tests for it.
