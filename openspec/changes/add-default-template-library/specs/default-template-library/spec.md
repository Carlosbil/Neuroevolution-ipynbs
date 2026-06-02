## ADDED Requirements

### Requirement: Expanded default template registry

The system SHALL provide a broader default set of known architecture templates
using only existing sequential, residual, and Inception Conv1D topology modes.

#### Scenario: Default templates include multiple families

- **WHEN** the template registry is queried
- **THEN** it includes sequential, residual, and Inception template families
- **AND** the default configured template ID list includes every default
  template in the registry.

#### Scenario: Sequential baselines are available

- **WHEN** configured templates are resolved
- **THEN** LeNet-like, AlexNet-like, and VGG-like Conv1D template IDs are
  available as sequential genomes.

#### Scenario: Residual variants are available

- **WHEN** configured templates are resolved
- **THEN** small, medium, and wide residual Conv1D template IDs are available.

#### Scenario: Inception variants are available

- **WHEN** configured templates are resolved
- **THEN** small, medium, and wide Inception Conv1D template IDs are available.

### Requirement: Default templates build valid models

The system SHALL convert every default template into a valid searchable genome
that can build and run an `EvolvableCNN` forward pass under supported
configuration bounds.

#### Scenario: Template factory validates every default template

- **WHEN** each default template ID is converted into a genome
- **THEN** the genome passes existing genome validation
- **AND** it includes template provenance and innovation genes.

#### Scenario: Template model construction succeeds

- **WHEN** an `EvolvableCNN` is created from each default template genome
- **THEN** the model builds without adding new PyTorch module families
- **AND** a forward pass over a small Conv1D input returns class logits.

#### Scenario: Runtime config clamps template size

- **WHEN** a default template is created under smaller configured layer or
  filter caps
- **THEN** the factory clamps the template genome through existing normalization
  and validation paths instead of returning an unsafe candidate.
