# Experiment Protocol Specification

## ADDED Requirements

### Requirement: Fold split isolation

Each experimental fold SHALL preserve three separate data subsets named `train`, `validation`, and `test`.

#### Scenario: Loading a fold for evolution

- **WHEN** a fold is loaded for evolutionary fitness evaluation
- **THEN** the loader contract SHALL expose separate train and validation loaders
- **AND** the test split SHALL NOT be concatenated with validation
- **AND** the test split SHALL NOT be iterated by the evolution-time training loop

### Requirement: Validation-only evolutionary fitness

Evolutionary fitness SHALL be computed only from validation metrics.

#### Scenario: Computing genome fitness

- **WHEN** a genome is evaluated across folds
- **THEN** each fold score SHALL be calculated from validation predictions
- **AND** the final genome fitness SHALL be the mean validation F1-score across folds
- **AND** no test prediction SHALL contribute to the fitness value

### Requirement: Validation-only checkpoint selection

Training checkpoints SHALL be selected using validation metrics only, and the default checkpoint metric SHALL match the evolutionary fitness metric.

#### Scenario: Selecting the best epoch state

- **WHEN** a model is trained for a fold
- **THEN** the best epoch state SHALL be chosen from validation F1-score by default
- **AND** test performance SHALL NOT be used for early stopping
- **AND** test performance SHALL NOT be used to select the saved global best checkpoint
- **AND** accuracy SHALL NOT be described as the primary checkpoint-selection metric unless the implementation and fitness metric are intentionally changed to match

### Requirement: Held-out final test evaluation

The test split SHALL be used only for final reporting after model selection is complete.

#### Scenario: Evaluating the selected genome

- **GIVEN** the best genome architecture has been selected using validation-derived fitness
- **WHEN** final evaluation is run for a fold
- **THEN** the model SHALL be trained on the training split
- **AND** validation SHALL be used for early stopping or best-state selection
- **AND** the restored validation-selected model SHALL be evaluated once on the test split
- **AND** reported final metrics SHALL come from the test split

### Requirement: Synthetic data split isolation

Synthetic samples SHALL be generated from training subjects only and SHALL NOT be present in validation or test splits.

#### Scenario: Preparing fold data with synthetic samples

- **WHEN** a fold is prepared for training, validation, and test
- **THEN** synthetic samples SHALL only be eligible for the training split
- **AND** validation and test splits SHALL contain real samples only
- **AND** synthetic samples derived from validation or test subjects SHALL NOT be included in training

### Requirement: Subject-level fold partitioning

Experimental folds SHALL be split at subject level before sample-level loading or augmentation.

#### Scenario: Assigning subjects to fold splits

- **WHEN** subjects are assigned to train, validation, and test subsets
- **THEN** every subject SHALL appear in only one subset within a fold
- **AND** the subject IDs assigned to each subset SHOULD be recorded or reproducible from the fold-generation configuration
- **AND** the documented split ratio SHALL match the generated fold structure

### Requirement: Accurate validation terminology

The article and documentation SHALL describe the protocol using terminology that matches the implemented fold design.

#### Scenario: Describing five fold evaluations

- **WHEN** the generated folds are fixed train/validation/test partitions rather than classical k-fold cross-validation
- **THEN** the method SHALL be described as five subject-level stratified hold-out partitions or an equivalent accurate term
- **AND** claims of classical cross-validation SHALL be avoided unless each subject is used in exactly one held-out test fold across the complete evaluation
