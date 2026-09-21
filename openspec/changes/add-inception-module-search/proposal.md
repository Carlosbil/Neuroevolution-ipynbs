# Add Inception Module Search

## Why

The current searchable model can build sequential Conv1D stacks and optional
residual Conv1D blocks. Both paths process each temporal scale in a single
chain. Audio problems often benefit from multi-scale temporal filters because
short transients and longer waveform patterns can both carry useful signal.

Inception-style modules, adapted from GoogLeNet to Conv1D, let one logical
convolutional stage evaluate several temporal receptive fields in parallel and
concatenate their outputs. Adding them to the genome gives evolution a richer
architecture search space for audio without replacing the existing algorithm.

## What Changes

- Add optional Inception Conv1D topology genes to each genome with
  backward-compatible defaults.
- Add a Conv1D Inception module with 1x1, reduced medium-kernel,
  reduced wide-kernel, and optional pooling-projection branches.
- Keep sequential Conv1D and residual Conv1D architectures valid and
  selectable. Inception and residual topology should be mutually exclusive in
  the first implementation.
- Update genome creation, validation, mutation, crossover, innovation tracking,
  speciation distance, architecture summaries, notebook output, and report
  utilities so Inception architectures are represented consistently.
- Include Inception fields in stable genome signatures used by evaluation
  caching.
- Make validation resource-aware by checking temporal dimensions, branch channel
  splits, and optional parameter-count limits before training.

## Scope

This change covers `test.ipynb` and the Python modules under
`neuroevolution/` that create, mutate, validate, build, evaluate, summarize,
or report genomes. It focuses on Inception-like Conv1D modules for the audio
architecture already used by `EvolvableCNN`.

## Non-goals

- Do not replace neuroevolution with a fixed GoogLeNet architecture.
- Do not add 2D image Inception models.
- Do not combine residual and Inception modules in the same genome yet.
- Do not introduce auxiliary classifiers from the original GoogLeNet paper.
- Do not change dataset loading semantics beyond any validation or reporting
  needed for Inception architectures.

## Success Criteria

- The system can generate sequential, residual, and Inception Conv1D genomes.
- Inception genomes build valid PyTorch modules and complete a forward pass on
  1D audio tensors.
- Branch outputs concatenate to the intended output channel count for each
  Inception module.
- Validation rejects Inception candidates with unsafe temporal dimensions,
  impossible branch splits, or excessive estimated parameter counts when a cap
  is configured.
- Mutation and crossover preserve complete, normalized Inception topology
  fields.
- Existing non-Inception genomes continue to build and evaluate as before.
- Smoke tests cover model construction, forward pass, validation, mutation,
  crossover, innovation genes, and summaries for Inception genomes.
