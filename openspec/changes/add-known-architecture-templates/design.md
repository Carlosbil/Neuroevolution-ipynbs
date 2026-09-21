## Context

The project already represents Conv1D architectures as mutable genome fields and
supports sequential, residual, and Inception topology modes. Initial population
creation currently uses mixed quartile depth caps plus a double-cap seed, while
mutation and crossover operate on normalized genome dictionaries with
innovation genes and structural history events.

Known architectures should therefore enter the system as genome templates, not
as fixed PyTorch models. A ResNet-like template can map to the existing residual
Conv1D fields, and a GoogLeNet-like template can map to the existing Inception
Conv1D fields. Once converted into a valid genome, the candidate must remain
fully evolvable.

## Goals / Non-Goals

**Goals:**

- Introduce a deterministic registry of known architecture templates that
  produces normalized genome dictionaries.
- Seed only a configurable portion of the initial population from templates.
- Support reusable template modules during mutation without bypassing existing
  validation, innovation tracking, resource checks, or cache signatures.
- Record template provenance in summaries and progress artifacts so results are
  interpretable.
- Keep random genome creation, quartile initialization, double-cap seeding,
  residual search, and Inception search compatible.

**Non-Goals:**

- Do not replace neuroevolution with fixed ResNet, GoogLeNet, or other static
  reference models.
- Do not add torchvision model imports or external architecture dependencies.
- Do not introduce 2D image architectures.
- Do not guarantee that a template survives mutation unchanged after entering
  the population.

## Decisions

### Template Registry

Add a module such as `neuroevolution/genetics/architecture_templates.py` with
small template descriptors and factory functions. Each template should include:

- `template_id`, for example `resnet_conv1d_small` or
  `googlenet_inception_conv1d_small`;
- human-readable `template_family`, such as `resnet` or `googlenet`;
- allowed `conv_topology`;
- optional depth, filter, kernel, residual, Inception, FC, optimizer, learning
  rate, dropout, and normalization defaults;
- a `source` flag that distinguishes `initial_seed` from `mutation_module`.

Factory functions should return ordinary genome dictionaries and then call
`validate_and_fix_genome`, `is_genome_valid`, and `build_innovation_genes`.
This keeps template-derived individuals inside the same safety envelope as
random genomes.

Alternative considered: instantiate fixed PyTorch reference networks and wrap
them in the evaluator. That would make mutation, crossover, validation,
speciation, and stable signatures much harder to keep consistent, so templates
should stay at the genome level.

### Configuration Controls

Add default config keys:

```python
"architecture_template_seed_fraction": 0.15,
"architecture_template_mutation_weight": 0.05,
"architecture_template_ids": [
    "resnet_conv1d_small",
    "googlenet_inception_conv1d_small",
],
"architecture_template_seed_min_random_fraction": 0.50,
"architecture_template_max_attempts": 20,
```

`architecture_template_seed_fraction` controls how many regular initial
population slots are template-derived. `architecture_template_seed_min_random_fraction`
is a guardrail: even if the seed fraction is configured too high, the
initializer must reserve at least this fraction for random/quartile individuals.

`validate_config` should ensure fractions are in `[0, 1]`, mutation weight is
in `[0, 1]`, template IDs are known and unique, and at least one random initial
slot remains when population size allows it.

Alternative considered: fold template probabilities into `conv_topology_weights`.
Topology weights only select sequential/residual/Inception mode; they do not
describe known depth/filter/kernel patterns or provenance, so a separate
template configuration is clearer.

### Initial Population Seeding

Extend `HybridNeuroevolution.initialize_population` so it calculates a template
seed quota for the regular population slots before the double-cap slot is
reserved. Template seeds should still respect quartile and incremental caps by
receiving the same capped config shape as random seeds, then being clamped and
validated.

Template seeds should be interleaved with random seeds rather than grouped at
the front of the population. Each seed records a structural event such as
`architecture_template_seed` with template ID, family, quartile, conv cap, FC
cap, and whether any fields were clamped.

Alternative considered: add template seeds after the random population is
complete. That risks exceeding population size or displacing the existing
double-cap behavior in unclear ways; computing a quota up front is simpler.

### Reusable Template Modules During Mutation

Extend mutation with a helper such as `_mutate_architecture_template_module`.
When selected by `architecture_template_mutation_weight`, the helper chooses a
compatible template module and applies a bounded structural change:

- switching to the template's topology package;
- replacing a contiguous Conv1D segment with template-inspired filter/kernel
  patterns;
- applying template residual or Inception fields where compatible;
- preserving existing FC layers and trainable hyperparameters unless the
  template explicitly provides safe alternatives.

The helper must clear cached evaluation fields, call centralized genome
normalization, reject invalid candidates, rebuild innovation genes, and append
an `architecture_template_module` structural event.

Alternative considered: add a crossover-only template donor parent. That would
require artificial parents and complicate selection accounting; a mutation
operator is more local and testable.

### Provenance, Signatures, and Distance

Add optional genome metadata:

```python
"architecture_template_id": None,
"architecture_template_family": None,
"architecture_template_origin": "random" | "initial_seed" | "mutation_module",
```

Template metadata should be included in progress JSON, reports, checkpoint
summaries, and architecture formatting. Stable genome signatures should include
template identity only when the template still affects structural fields or
when provenance is used for cache separation. If two genomes normalize to the
same architecture and hyperparameters, reusing cached fitness is acceptable
unless the implementation deliberately treats provenance as a different
experimental condition.

Speciation may include a small provenance component, but topology, innovation
genes, and hyperparameters remain the primary distance signals. This avoids
over-separating a template-derived genome after it has evolved away from its
source.

## Risks / Trade-offs

- Template seeding reduces search novelty -> keep a minimum random seed
  fraction and low default seed fraction.
- Template modules can produce genomes that exceed depth, channel, or parameter
  limits -> run the same validation and parameter estimation used for random
  genomes before admitting candidates.
- Provenance can pollute cache behavior if included too aggressively -> base
  cache signatures on normalized architecture fields by default and document
  any provenance-based separation.
- ResNet and GoogLeNet names can imply exact paper architectures -> label
  templates as Conv1D adaptations and avoid claiming equality with image
  reference models.
- Existing initialization already reserves a double-cap slot -> compute
  template quotas only for regular slots so the exploratory double-cap seed
  remains available.
