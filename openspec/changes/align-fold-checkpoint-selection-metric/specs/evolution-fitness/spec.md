# Evolution Fitness Specification

## ADDED Requirements

### Requirement: Fold checkpoint selection uses configurable objective metric

The system SHALL select the best fold model state using a configurable objective
metric instead of hard-coded accuracy.

#### Scenario: F1 selection preserves the F1-best epoch

- **GIVEN** `fold_selection_metric` resolves to `f1_score`
- **AND** one epoch has higher accuracy but lower F1 than another epoch
- **WHEN** fold training evaluates candidate epochs
- **THEN** the fold best state is selected from the epoch with the higher F1
- **AND** the selected state is the one returned for checkpointing.

#### Scenario: Accuracy selection remains available explicitly

- **GIVEN** `fold_selection_metric` is configured as `accuracy`
- **WHEN** fold training evaluates candidate epochs
- **THEN** the fold best state is selected by accuracy
- **AND** behavior remains compatible with the previous accuracy-based
  selection when explicitly requested.

#### Scenario: Selection metric defaults to fitness objective

- **GIVEN** `fold_selection_metric` is configured as `fitness_metric`
- **AND** `fitness_metric` is configured as `f1_score`
- **WHEN** fold training resolves the selection metric
- **THEN** it uses F1-score for best-state selection.

### Requirement: Selection metric is visible in metrics and logs

The system SHALL record which metric selected the best fold state and what
score was selected.

#### Scenario: Fold metrics include selection metadata

- **GIVEN** a fold has completed training and evaluation
- **WHEN** fold metrics are returned
- **THEN** the metrics include `selection_metric`
- **AND** the metrics include `best_selection_score`
- **AND** the metrics include `best_epoch` when epoch information is available.

#### Scenario: Logs avoid hard-coded accuracy wording

- **GIVEN** the configured selection metric is not accuracy
- **WHEN** training or final evaluation logs best-state selection
- **THEN** the logs name the configured selection metric
- **AND** do not describe the selected state as accuracy-selected.

### Requirement: Metric configuration is validated

The system SHALL reject unsupported objective metric names before a long
evolution run starts.

#### Scenario: Unsupported selection metric fails validation

- **GIVEN** config contains an unknown `fold_selection_metric`
- **WHEN** `validate_config` runs
- **THEN** it raises a clear validation error.

#### Scenario: Recall aliases sensitivity

- **GIVEN** config uses `recall` as a metric name
- **WHEN** the metric is resolved or read from computed metrics
- **THEN** it maps to the same value as `sensitivity`.

### Requirement: Final evaluation uses the same selection metric

The system SHALL use the same configurable model-selection metric during final
5-fold evaluation.

#### Scenario: Final evaluation selects best epoch by configured metric

- **GIVEN** final evaluation is training a fold
- **AND** `fold_selection_metric` resolves to `f1_score`
- **WHEN** periodic validation evaluates epochs
- **THEN** the best epoch is selected by F1-score
- **AND** final fold metadata records F1 as the selection metric.
