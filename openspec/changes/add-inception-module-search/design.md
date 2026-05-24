# Design: Inception Module Search

## Current State

`EvolvableCNN` currently supports two Conv1D topology modes:

- Sequential Conv1D layers, where each convolution is followed by
  normalization, activation, pooling, and optional dropout.
- Residual Conv1D blocks, where grouped convolutional units use an identity or
  1x1 projection shortcut and pool at the block boundary.

Genome creation, validation, mutation, crossover, innovation tracking, and
speciation already understand the base layer fields and residual fields. The
Inception work should follow the same pattern: add topology fields with safe
defaults, normalize them centrally, and keep legacy genomes valid.

## Proposed Genome Fields

Add Inception fields with defaults that preserve old genomes:

```python
inception_enabled = genome.get("inception_enabled", False)
inception_reduction_ratio = genome.get("inception_reduction_ratio", 0.5)
inception_pool_branch = genome.get("inception_pool_branch", True)
```

Add config values:

```python
"conv_topology_weights": {
    "sequential": 0.45,
    "residual": 0.30,
    "inception": 0.25,
},
"inception_reduction_ratio_options": [0.25, 0.5],
"inception_pool_branch_options": [True, False],
"inception_min_branch_channels": 1,
"inception_mutation_weight": 0.15,
```

`conv_topology_weights` should be preferred for new genome creation. Existing
residual weights can remain supported as a fallback for older config snippets.

Inception and residual topology are mutually exclusive for this change. A
normalization helper should resolve the active topology in this order:

1. legacy genomes with no new fields remain sequential;
2. residual-enabled genomes remain residual unless Inception is explicitly
   enabled by a new mutation or new genome creation;
3. if both `residual_enabled` and `inception_enabled` are true, keep one
   topology and normalize the other to false using the selected topology source
   from creation, mutation, or crossover.

## Inception Conv1D Module

Introduce `InceptionConv1DModule` in
`neuroevolution/models/evolvable_cnn.py`.

Each logical module consumes one existing Conv1D gene index and uses:

- `filters[i]` as the total output channel count after concatenation.
- `kernel_sizes[i]` as the wide temporal kernel, normalized to an odd value and
  at least 5 in Inception mode.
- a medium branch with kernel size 3.
- `inception_reduction_ratio` to determine the 1x1 reduction channels before
  medium and wide branches.
- `inception_pool_branch` to include or remove the pooling-projection branch.

The module branches are:

1. 1x1 Conv1D unit.
2. 1x1 reduction Conv1D unit followed by a same-padded kernel-3 Conv1D unit.
3. 1x1 reduction Conv1D unit followed by a same-padded wide-kernel Conv1D unit.
4. Optional stride-1 MaxPool1D with same-length padding followed by a 1x1
   projection Conv1D unit.

Branch outputs are concatenated along the channel dimension. A deterministic
channel-splitting helper should allocate branch output channels so the sum is
exactly `filters[i]`, with at least `inception_min_branch_channels` per active
branch. If the requested total channel count is too small, validation should
fix it within `max_filters` or reject the genome.

The temporal dimension must remain unchanged inside the module. As in the
sequential path, apply `MaxPool1d(2, 2)` after each logical module, followed by
dropout except where the existing code omits it.

## Model Construction

Extend `_build_conv_layers` so it selects one path:

- sequential path when neither residual nor Inception is active;
- residual path when `residual_enabled=True`;
- Inception path when `inception_enabled=True`.

The Inception path should build one `InceptionConv1DModule` per logical
convolutional layer. `in_channels` becomes the concatenated output channels of
the previous module, which is `filters[i]`.

Normalization should reuse existing behavior. For Conv1D tensors, layer
normalization should use the existing channel-aware helper rather than plain
`nn.LayerNorm` on `(N, C, L)` tensors.

## Validation

Centralize Inception normalization in `validate_and_fix_genome`, parallel to
the residual normalization already present.

Validation should enforce:

- `inception_enabled` is a boolean and defaults to false.
- `inception_reduction_ratio` is one of the configured options.
- `inception_pool_branch` is one of the configured options.
- residual and Inception modes are not active at the same time.
- Inception wide kernels are odd and at least 5.
- each Inception module can allocate the requested active branch channels.
- expected temporal length after pooling stays above the minimum safe length.
- estimated model parameters are below `max_model_parameters` when configured.

Because each Inception module maps to one logical Conv1D layer and pools once
after the module, its pooling count is:

```python
pool_count = num_conv_layers
```

Residual mode keeps its existing block-level pooling behavior.

Parameter estimation should account for all branch convolutions and projection
layers without instantiating a PyTorch model in normal validation paths.

## Genetic Operators

Genome creation should sample one topology from `conv_topology_weights`.
Sequential, residual, and Inception genomes should all pass through the same
normalization and validity checks before returning.

Mutation should be able to:

- switch the active topology to or from Inception;
- change `inception_reduction_ratio`;
- toggle `inception_pool_branch` when multiple options are configured;
- keep residual fields valid when the genome leaves residual mode.

Crossover should copy a complete topology package from one parent or choose a
single topology source deterministically for each child. Children must not
inherit partial states such as both residual and Inception enabled.

Innovation tracking should include:

- `inception_enabled`
- `inception_reduction_ratio`
- `inception_pool_branch`

Speciation distance should include a small topology component so sequential,
residual, and Inception lineages are not treated as identical when their base
layer lists match.

## Stable Genome Signature

Any cache signature for equivalent architectures must include Inception fields:

```python
signature = (
    genome["num_conv_layers"],
    tuple(genome["filters"]),
    tuple(genome["kernel_sizes"]),
    genome["num_fc_layers"],
    tuple(genome["fc_nodes"]),
    tuple(genome["activations"]),
    genome["dropout_rate"],
    genome["learning_rate"],
    genome["optimizer"],
    genome["normalization_type"],
    genome.get("residual_enabled", False),
    genome.get("residual_block_size", 2),
    genome.get("residual_projection", "auto"),
    genome.get("inception_enabled", False),
    genome.get("inception_reduction_ratio", 0.5),
    genome.get("inception_pool_branch", True),
)
```

## Notebook and Reporting

`test.ipynb` should expose the Inception config keys near the architecture
search configuration. Final best-genome summaries, checkpoint summaries,
architecture summaries, progress JSON, and report utilities should state:

- active Conv1D topology;
- Inception enabled or disabled;
- reduction ratio;
- pool branch enabled or disabled;
- branch kernels and output-channel split for Inception modules.

## Risks and Mitigations

- **Risk:** Inception modules increase parameters and memory usage.
  **Mitigation:** use reduction branches, validate branch channel counts, and
  respect `max_model_parameters` when configured.
- **Risk:** Inception and residual fields can conflict.
  **Mitigation:** normalize to a single active topology in one helper used by
  creation, mutation, crossover, validation, and model construction.
- **Risk:** Small filter counts cannot be split across branches.
  **Mitigation:** fix totals upward when possible or reject the genome before
  training.
- **Risk:** Existing `kernel_sizes` may be ambiguous in Inception mode.
  **Mitigation:** define it as the wide branch kernel and document that kernel
  3 is always the medium branch.

## GoogLeNet Comparison

This design keeps the core Inception v1 ideas and adapts them to audio:

- parallel branches process the same input tensor;
- 1x1 convolutions provide cheap direct features and dimensionality reduction;
- medium and wide convolutions capture multiple receptive-field scales;
- a pooling branch preserves a non-convolutional local summary path;
- branch outputs concatenate along the channel dimension.

It intentionally does not copy the full image model. It uses Conv1D, keeps the
existing neuroevolution training loop, does not add auxiliary classifiers, and
does not impose fixed stage depths.
