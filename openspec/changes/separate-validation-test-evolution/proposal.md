# Separate Validation and Test During Evolution

## Why

The current fitness evaluation loads `X_val/y_val` and `X_test/y_test` for each
fold, concatenates both splits, and uses the combined data while evolution is
selecting genomes. This gives the search process more evaluation samples, but
it also lets evolutionary pressure optimize indirectly against the test split.

For scientific validity, model selection should be driven by validation data.
The test split should remain untouched until the final evaluation of the
selected best genome. This change makes the experimental protocol cleaner and
keeps reported test metrics meaningful.

## What Changes

- Change evolutionary fitness so training uses each fold's training split and
  validation/early-stopping/fitness use only each fold's validation split.
- Reserve each fold's test split for final evaluation only.
- Make final 5-fold evaluation load train, validation, and test splits
  separately so it can select/checkpoint by validation performance and report
  final metrics on test.
- Rename or clarify loader variables and logs so `validation` and `test` are
  not conflated.
- Add tests that prevent regression to `val + test` concatenation during
  evolution.
- Update user-facing documentation/messages that currently describe the final
  evaluation as using the same combined methodology as evolution.

## Scope

This change covers the evaluation protocol in:

- `neuroevolution/evolution/fitness.py`
- `neuroevolution/evaluation/cross_validation.py`
- relevant tests under `tests/`
- README or improvement docs where they describe fitness/final evaluation
  semantics

## Non-goals

- Do not introduce multi-fidelity evaluation in this change.
- Do not change the genetic operators, model architectures, residual search, or
  Inception search.
- Do not change the physical fold file format.
- Do not alter the target metric from F1 in this change, except where naming
  and reporting must clarify whether metrics are validation or test metrics.
- Do not remove final 5-fold reporting.

## Success Criteria

- During evolution, `evaluate_fitness` never uses `X_test/y_test` for
  validation, early stopping, checkpoint selection, or fitness aggregation.
- The final evaluation uses validation data for model selection and reports
  metrics on test data.
- Logs and metric dictionaries make it clear which split produced each metric.
- Existing training behavior remains backward-compatible for callers that only
  need train/eval loaders, while new code can request validation-only or
  train/validation/test loaders explicitly.
- Tests fail if evolution reintroduces `val + test` concatenation.
