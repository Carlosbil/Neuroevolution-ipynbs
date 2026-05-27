# Evolution Fitness Specification

## ADDED Requirements

### Requirement: Evolution fitness uses validation split only

The system SHALL use validation data, not test data, when computing fitness
during evolutionary search.

#### Scenario: Fold fitness is validation-only

- **GIVEN** a fold with separate train, validation, and test files
- **WHEN** an individual is evaluated during evolution
- **THEN** gradient updates use only the train split
- **AND** epoch validation uses only the validation split
- **AND** the fold score used for fitness is computed only from the validation
  split.

#### Scenario: Test split is not concatenated into evolutionary evaluation

- **GIVEN** validation and test splits with different sample counts or labels
- **WHEN** the evolution fold evaluation loader is built
- **THEN** the loader contains validation samples only
- **AND** no `val + test` concatenation is used for evolutionary fitness.

#### Scenario: Evolution metrics identify their source split

- **GIVEN** an individual has completed evolutionary fitness evaluation
- **WHEN** metrics are returned or logged
- **THEN** the metrics identify validation as the split used for fitness
- **AND** logs do not label validation fitness metrics as test metrics.

### Requirement: Test split is reserved for final evaluation

The system SHALL keep the test split out of model selection and use it only for
final reporting after a best genome has been selected.

#### Scenario: Final evaluation selects epoch by validation

- **GIVEN** the best genome is being evaluated after evolution
- **WHEN** each fold is trained for final evaluation
- **THEN** the train split is used for gradient updates
- **AND** the validation split is used to select or restore the best epoch.

#### Scenario: Final reported metrics are test-only

- **GIVEN** the final evaluation has restored the best validation-selected model
  for a fold
- **WHEN** final metrics are computed
- **THEN** accuracy, sensitivity, specificity, precision, F1, and AUC are
  computed on the test split only
- **AND** aggregate final metrics are derived from test-only fold metrics.

#### Scenario: Loader cache respects split mode

- **GIVEN** the same fold is requested once for validation evaluation and once
  for test evaluation
- **WHEN** DataLoader caching is enabled
- **THEN** validation and test requests use distinct cache keys
- **AND** a cached test loader cannot be returned for an evolutionary
  validation request.
