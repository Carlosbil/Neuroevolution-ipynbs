## ADDED Requirements

### Requirement: Known architecture template registry

The system SHALL provide a controlled registry of known architecture templates
that can produce valid Conv1D genome dictionaries compatible with the existing
evolution pipeline.

#### Scenario: Template registry exposes supported templates

- **WHEN** template seeding or template mutation requests the available
  templates
- **THEN** the system returns only configured template IDs that exist in the
  registry
- **AND** each template includes a stable ID, family name, supported topology,
  and genome construction metadata.

#### Scenario: Template becomes a normalized genome

- **WHEN** a registered template is converted into an individual
- **THEN** the output is a normal genome dictionary accepted by existing genome
  validation
- **AND** the genome includes innovation genes and structural history.

#### Scenario: Unsupported template is configured

- **WHEN** configuration references an unknown architecture template ID
- **THEN** configuration validation fails before evolution starts.

### Requirement: Controlled initial population seeding

The system SHALL seed only a configurable and bounded portion of the initial
population from known architecture templates.

#### Scenario: Initial population uses configured template fraction

- **WHEN** population initialization starts with template seeding enabled
- **THEN** the initializer creates no more template-derived individuals than the
  configured template seed fraction allows
- **AND** the remaining initial slots are filled through the existing random
  quartile initialization process.

#### Scenario: Minimum random exploration is preserved

- **WHEN** the configured template seed fraction would consume too many initial
  slots
- **THEN** the initializer clamps the template seed count so the configured
  minimum random fraction remains available
- **AND** the double-cap exploratory seed remains reserved when population size
  allows it.

#### Scenario: Template seed respects current caps

- **WHEN** a template-derived initial individual is created for a quartile or
  incremental complexity cap
- **THEN** its convolutional and fully connected layer counts are clamped to the
  active caps
- **AND** validation rejects or fixes unsafe dimensions before the individual
  enters the population.

### Requirement: Reusable template module mutation

The system SHALL support a bounded mutation operator that can insert or apply
known architecture template modules to existing genomes.

#### Scenario: Template mutation is selected

- **WHEN** mutation samples the architecture template mutation operator
- **THEN** the operator chooses a configured template compatible with the genome
  and active search configuration
- **AND** it applies a bounded topology or Conv1D segment change rather than
  replacing the whole evolutionary process.

#### Scenario: Template module mutation keeps genome valid

- **WHEN** a template module mutation modifies a genome
- **THEN** the resulting genome is normalized, validated, assigned a fresh ID,
  and given rebuilt innovation genes
- **AND** invalid template-derived candidates are retried or rejected before
  being returned.

#### Scenario: Template mutation records structural provenance

- **WHEN** a reusable template module changes a genome
- **THEN** the genome structural history records the template ID, template
  family, mutation origin, and affected Conv1D segment or topology fields.

### Requirement: Template-derived genome provenance

The system SHALL retain enough provenance to distinguish template-derived
genomes from purely random genomes in logs, summaries, and saved artifacts.

#### Scenario: Template seed provenance is stored

- **WHEN** a genome is created from a known architecture template
- **THEN** the genome records template ID, template family, and origin as an
  initial seed
- **AND** progress JSON and checkpoint artifacts can serialize those fields.

#### Scenario: Random genome provenance remains explicit

- **WHEN** a genome is created without a known architecture template
- **THEN** template provenance fields are absent or normalized to random/none
- **AND** existing random genome behavior remains unchanged.

#### Scenario: Reports include template provenance

- **WHEN** architecture summaries, final best-genome reports, checkpoint
  summaries, or compact architecture formatting display a template-derived
  genome
- **THEN** the output identifies the template ID and family without hiding the
  actual evolved layer counts, filters, kernels, topology, and hyperparameters.

### Requirement: Template-derived genomes remain evolvable

The system SHALL treat template-derived genomes as normal evolutionary
individuals after they enter the population.

#### Scenario: Crossover can use template-derived parents

- **WHEN** crossover receives one or more template-derived parents
- **THEN** child genomes are produced through the existing crossover path
- **AND** complete topology packages, innovation genes, and template provenance
  are normalized consistently.

#### Scenario: Standard mutation can alter template-derived genomes

- **WHEN** a template-derived genome is selected for standard mutation
- **THEN** normal layer, topology, filter, kernel, FC, dropout, optimizer, and
  learning-rate mutations remain available
- **AND** mutation is not constrained to preserve the original template exactly.

#### Scenario: Fitness caching remains architecture-safe

- **WHEN** stable signatures are computed for template-derived genomes
- **THEN** signatures distinguish genomes whose normalized architecture,
  topology, or hyperparameters differ
- **AND** cache reuse is not blocked solely because a genome has template
  provenance when the normalized searchable architecture is identical.
