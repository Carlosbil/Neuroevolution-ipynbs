# Evolvable CNN Specification

## ADDED Requirements

### Requirement: Searchable Inception Conv1D topology

The system SHALL represent Inception Conv1D topology explicitly in each genome
using backward-compatible defaults when fields are absent.

#### Scenario: Default non-Inception genome remains valid

- **GIVEN** a genome created before Inception support exists
- **WHEN** the genome is validated or passed to `EvolvableCNN`
- **THEN** missing Inception fields are treated as Inception disabled
- **AND** the existing sequential or residual Conv1D behavior is preserved.

#### Scenario: Genome has one active Conv1D topology

- **GIVEN** a genome with topology fields for sequential, residual, and
  Inception modes
- **WHEN** the genome is created, mutated, crossed over, or validated
- **THEN** no more than one non-sequential topology is active
- **AND** residual and Inception modes are not enabled at the same time.

#### Scenario: Inception genome has stable topology fields

- **GIVEN** an Inception-enabled genome
- **WHEN** the genome is created, mutated, crossed over, serialized, or logged
- **THEN** it includes Inception topology fields such as
  `inception_enabled`, `inception_reduction_ratio`, and
  `inception_pool_branch`
- **AND** those fields are normalized to supported values.

### Requirement: Inception Conv1D module construction

The system SHALL build Inception Conv1D modules that process 1D audio tensors
through parallel temporal branches and concatenate branch outputs.

#### Scenario: Branches preserve temporal length before concatenation

- **GIVEN** an Inception Conv1D module with same-padded branches
- **WHEN** the 1x1, medium-kernel, wide-kernel, and optional pooling-projection
  branches run
- **THEN** each branch output has the same temporal length
- **AND** branch outputs can be concatenated on the channel dimension.

#### Scenario: Concatenated channels match genome filters

- **GIVEN** an Inception module whose genome layer has `filters[i]`
- **WHEN** branch output channels are allocated
- **THEN** the sum of active branch output channels equals `filters[i]`
- **AND** each active branch receives at least the configured minimum channel
  count.

#### Scenario: Module-level pooling follows concatenation

- **GIVEN** an Inception-enabled genome
- **WHEN** a logical Inception module completes its branch concatenation
- **THEN** pooling is applied after the module boundary
- **AND** the next module receives the concatenated output channels.

### Requirement: Resource-aware Inception validation

The system SHALL permit Inception candidates only when validation predicts safe
temporal dimensions, valid branch channel splits, and configured resource
limits.

#### Scenario: Temporal dimensions remain safe

- **GIVEN** an Inception genome with `num_conv_layers`
- **WHEN** the validator estimates the final temporal length
- **THEN** it bases downsampling on one pooling operation per Inception module
- **AND** rejects architectures whose predicted temporal dimension is below the
  required minimum.

#### Scenario: Branch split is impossible

- **GIVEN** an Inception genome whose requested filter count cannot be split
  across active branches
- **WHEN** the genome is validated
- **THEN** validation fixes the filter count within configured limits or rejects
  the genome before training starts.

#### Scenario: Parameter cap includes Inception branches

- **GIVEN** `max_model_parameters` is configured
- **WHEN** an Inception genome is validated
- **THEN** estimated parameters include all branch convolutions and projection
  layers
- **AND** validation rejects candidates above the cap.

### Requirement: Genetic operators understand Inception genes

The system SHALL include Inception topology in genome creation, mutation,
crossover, innovation tracking, speciation distance, and evaluation signatures.

#### Scenario: Creation samples Inception topology

- **GIVEN** topology weights that allow Inception
- **WHEN** random genome creation samples a new genome
- **THEN** it may produce a complete Inception-enabled genome
- **AND** the genome passes normal validation before entering the population.

#### Scenario: Mutation changes Inception topology

- **GIVEN** a genome selected for mutation
- **WHEN** Inception topology mutation is applied
- **THEN** the operator may enable or disable Inception or adjust supported
  Inception parameters
- **AND** it records a structural event for the change.

#### Scenario: Crossover combines topology consistently

- **GIVEN** parents with different Conv1D topology fields
- **WHEN** crossover creates children
- **THEN** each child receives a complete, valid topology
- **AND** residual and Inception fields remain mutually exclusive.

#### Scenario: Stable signature distinguishes Inception architectures

- **GIVEN** two genomes that are identical except for Inception topology
- **WHEN** a stable genome signature is computed
- **THEN** the signatures differ so cached fitness is not reused incorrectly.

### Requirement: Inception architecture visibility

The system SHALL expose Inception topology in notebook output, summaries,
reports, checkpoints, and saved artifacts.

#### Scenario: Best genome summary includes Inception fields

- **GIVEN** the best genome after evolution
- **WHEN** `test.ipynb`, `get_architecture_summary`, checkpoint summaries, or
  report utilities display the architecture
- **THEN** the output states whether Inception modules were used
- **AND** shows the Inception reduction ratio, pool branch setting, branch
  kernels, and channel split.
