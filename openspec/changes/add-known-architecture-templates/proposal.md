## Why

The current search relies on randomly generated or incrementally mutated
architectures, which can spend many evaluations rediscovering patterns already
known to train well. Controlled architecture templates can shorten the search by
starting from proven Conv1D adaptations of designs such as ResNet or GoogLeNet
while still allowing evolution to mutate architecture and hyperparameters.

## What Changes

- Add a controlled registry of known architecture templates that can be adapted
  to the existing Conv1D genome format.
- Allow templates to seed a configurable portion of the initial population.
- Allow reusable template modules to be inserted during mutation when their
  topology is compatible with the active search configuration.
- Keep fully generated genomes available so the search does not collapse into a
  fixed set of hand-picked architectures.
- Validate, normalize, log, summarize, and cache template-derived genomes with
  enough provenance to distinguish them from purely random candidates.

## Capabilities

### New Capabilities

- `architecture-template-seeding`: Controlled use of known architecture
  templates as initial individuals or reusable modules inside the evolutionary
  search process.

### Modified Capabilities

None.

## Impact

This affects genome creation, mutation, validation, stable genome signatures,
architecture summaries, logging, reporting, and `test.ipynb` configuration. It
does not add external dependencies or replace the neuroevolution process with
fixed ResNet or GoogLeNet models.
