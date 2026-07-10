# Fix Train/Validation/Test Protocol

## Summary

Correct the experimental pipeline so that each fold keeps three separate subsets throughout the full workflow:

- `train`: used only for model optimization.
- `validation`: used for epoch monitoring, checkpoint selection, and evolutionary fitness.
- `test`: used only once for final reporting after architecture/model selection is complete.

The current implementation loads `train`, `validation`, and `test` files, but then concatenates `validation + test` into the evaluation loader used during evolution and final evaluation. This leaks test information into early stopping, checkpoint selection, fitness computation, and reported results.

## Why

The article checklist requires a strict train/validation/test protocol. The current code does not satisfy that protocol:

- Test samples are evaluated during training epochs.
- Test samples influence per-fold best-state selection.
- Test samples influence evolutionary fitness.
- The final evaluation reuses the same combined validation/test loader.

This invalidates claims that the test set is an independent held-out evaluation set. Correcting this is necessary before reporting final scientific results.

## Goals

- Preserve the existing fold files and dataset layout.
- Introduce split-aware data loading that never combines validation and test.
- Use validation metrics only for fitness and checkpoint selection.
- Rework final evaluation so test metrics are computed only after model selection.
- Align README/notebook wording with the corrected protocol.
- Add tests that prevent future test leakage.

## Non-Goals

- Regenerating fold `.npy` files.
- Changing the subject-level split generation procedure.
- Re-running full experiments or updating article result tables.
- Redesigning the neuroevolution algorithm beyond the split protocol.

## Impact

Primary code paths:

- `neuroevolution/evolution/fitness.py`
- `neuroevolution/evaluation/cross_validation.py`
- `neuroevolution/data/loader.py`
- README and notebook-facing documentation

Expected behavior after the change:

```text
During evolution:
train -> gradient updates
validation -> epoch monitoring, best state, fitness, global checkpoint
test -> not loaded or evaluated

Final evaluation:
train -> gradient updates for selected architecture
validation -> final run checkpoint/early stopping
test -> final metrics only
```
