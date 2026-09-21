# Add Residual Block Search

## Why

The current searchable model builds a purely sequential Conv1D stack. The
evolution engine can already explore deeper candidates through layer-count
mutation, quartile initialization, and the double-cap seed, but deeper
sequential stacks still suffer from harder optimization and aggressive spatial
downsampling.

Residual blocks, in the style of ResNet, let deeper networks preserve an
identity path through the convolutional stack. This gives the search process a
better chance to train larger architectures when GPU memory and sequence length
make them viable.

## What Changes

- Add residual Conv1D blocks as explicit genes in the architecture search
  space.
- Keep existing sequential Conv1D architectures as the default/backward
  compatible path.
- Update genome creation, validation, mutation, crossover, innovation tracking,
  speciation distance, architecture summaries, and notebook output so residual
  architectures are represented consistently.
- Make residual depth resource-aware: validation must account for block-level
  pooling and must reject architectures that would produce invalid dimensions or
  exceed configured safety limits.
- Include residual fields in any stable genome signature used for evaluation
  caching.

## Scope

This change covers `test.ipynb` and the Python scripts under
`neuroevolution/` that create, mutate, validate, build, evaluate, summarize,
or report genomes. It focuses on Conv1D residual blocks for the audio
architecture already used by `EvolvableCNN`.

## Non-goals

- Do not replace the neuroevolution algorithm with a fixed ResNet.
- Do not introduce 2D image ResNet models.
- Do not require residual blocks for every genome.
- Do not change dataset loading semantics beyond any validation/reporting
  needed for residual architectures.

## Success Criteria

- The system can generate both sequential and residual Conv1D genomes.
- Residual genomes build valid PyTorch modules with identity or projection
  shortcuts as needed.
- Deep residual candidates are allowed when their computed dimensions and
  configured resource limits are safe.
- Mutation and crossover preserve or alter residual topology without producing
  inconsistent genome fields.
- Existing non-residual genomes continue to build and evaluate as before.
- Smoke tests cover model construction, forward pass, validation, mutation, and
  crossover for residual genomes.
