# Notebook Held-Out Evaluation Specification

## ADDED Requirements

### Requirement: Notebook invokes final held-out evaluation

`test.ipynb` SHALL invoke final five-fold evaluation after evolutionary selection has produced a best genome.

#### Scenario: Evolution completes successfully

- **GIVEN** `best_genome` was selected through validation-derived evolutionary fitness
- **WHEN** the notebook reaches the post-evolution evaluation step
- **THEN** it SHALL call `evaluate_5fold_cross_validation` with that genome, the configured experiment settings, and the selected device
- **AND** it SHALL retain the returned held-out results separately from the evolutionary metrics

### Requirement: Final results are persisted with split provenance

Final held-out evaluation results SHALL be serialized inside the configured artifacts directory.

#### Scenario: At least one final evaluation fold succeeds

- **WHEN** the evaluator aggregates final fold results
- **THEN** it SHALL write a timestamped JSON file under `config['artifacts_dir']`
- **AND** the returned results SHALL identify the persisted file path
- **AND** the serialized result SHALL identify `validation` as the selection split and `test` as the evaluation split

### Requirement: Notebook presents article-ready held-out metrics

The notebook SHALL present final test results independently of validation-selection metrics.

#### Scenario: Rendering successful final evaluation results

- **WHEN** one or more folds produce held-out results
- **THEN** the notebook SHALL show metrics for every successful fold
- **AND** it SHALL show mean and standard deviation for accuracy, sensitivity, specificity, F1, and AUC
- **AND** it SHALL show one labelled confusion matrix for each successful fold
- **AND** it SHALL label these outputs as held-out test evaluation
