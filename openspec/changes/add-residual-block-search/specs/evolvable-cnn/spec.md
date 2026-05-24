# Evolvable CNN Specification

## ADDED Requirements

### Requirement: Searchable residual Conv1D topology

The system SHALL represent residual Conv1D topology explicitly in each genome
using backward-compatible defaults when fields are absent.

#### Scenario: Default sequential genome remains valid

- **GIVEN** a genome created before residual support exists
- **WHEN** the genome is validated or passed to `EvolvableCNN`
- **THEN** missing residual fields are treated as residual disabled
- **AND** the existing sequential Conv1D behavior is preserved.

#### Scenario: Residual genome has stable topology fields

- **GIVEN** a residual-enabled genome
- **WHEN** the genome is created, mutated, crossed over, serialized, or logged
- **THEN** it includes residual topology fields such as
  `residual_enabled`, `residual_block_size`, and `residual_projection`
- **AND** those fields are normalized to supported values.

### Requirement: Residual block model construction

The system SHALL build residual Conv1D blocks that add a shortcut path to the
block output only when tensor shapes are compatible or can be made compatible
with an automatic projection.

#### Scenario: Channels match

- **GIVEN** a residual block whose input and output channel counts match
- **WHEN** the model performs a forward pass
- **THEN** the shortcut path uses identity mapping
- **AND** the residual addition succeeds without shape mismatch.

#### Scenario: Channels differ

- **GIVEN** a residual block whose input and output channel counts differ
- **WHEN** `residual_projection` is `auto`
- **THEN** the shortcut path uses a 1x1 Conv1D projection
- **AND** the residual addition succeeds without shape mismatch.

#### Scenario: Spatial length is preserved inside a block

- **GIVEN** a residual block with odd kernel sizes and same-padding Conv1D
- **WHEN** all convolutions inside the block run
- **THEN** the temporal dimension is unchanged until the block-level pooling
  step.

### Requirement: Resource-aware deep residual search

The system SHALL permit deeper residual candidates only when validation predicts
safe temporal dimensions and configured resource limits.

#### Scenario: Block-level pooling changes safe depth

- **GIVEN** residual mode with a block size greater than one
- **WHEN** the validator estimates the final temporal length
- **THEN** it bases downsampling on the number of pooling operations rather than
  raw convolution layer count.

#### Scenario: Architecture is too deep for the input length

- **GIVEN** a residual genome whose predicted temporal dimension falls below
  the required minimum
- **WHEN** the genome is validated
- **THEN** validation rejects the genome before training starts.

### Requirement: Genetic operators understand residual genes

The system SHALL include residual topology in genome creation, mutation,
crossover, innovation tracking, speciation distance, and evaluation signatures.

#### Scenario: Mutation changes residual topology

- **GIVEN** a genome selected for mutation
- **WHEN** residual mutation is applied
- **THEN** the operator may toggle residual mode or adjust supported block
  parameters
- **AND** it records a structural event for the change.

#### Scenario: Crossover combines residual topology consistently

- **GIVEN** two parents with residual topology fields
- **WHEN** crossover creates children
- **THEN** each child receives a complete, valid residual topology
- **AND** residual innovation genes are rebuilt for the child.

#### Scenario: Stable signature distinguishes residual architectures

- **GIVEN** two genomes that are identical except for residual topology
- **WHEN** a stable genome signature is computed
- **THEN** the signatures differ so cached fitness is not reused incorrectly.

### Requirement: Residual architecture visibility

The system SHALL expose residual topology in notebook output, summaries,
reports, and saved artifacts.

#### Scenario: Best genome summary includes residual fields

- **GIVEN** the best genome after evolution
- **WHEN** `test.ipynb`, `get_architecture_summary`, or report utilities display
  the architecture
- **THEN** the output states whether residual blocks were used and shows the
  residual block parameters.
