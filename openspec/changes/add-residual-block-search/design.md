# Design: Residual Block Search

## Current State

`EvolvableCNN` currently builds a flat `ModuleList` of Conv1D, normalization,
activation, pooling, and dropout layers. Every convolution is followed by a
pooling operation, so validation estimates safe depth as
`sequence_length / (2 ** num_conv_layers)`.

Genome creation, validation, mutation, crossover, innovation tracking, and
speciation are keyed around these fields:

- `num_conv_layers`
- `filters`
- `kernel_sizes`
- `num_fc_layers`
- `fc_nodes`
- `activations`
- `dropout_rate`
- `learning_rate`
- `optimizer`
- `normalization_type`

## Proposed Genome Fields

Add residual topology fields with defaults that preserve old genomes:

```python
residual_enabled = genome.get("residual_enabled", False)
residual_block_size = genome.get("residual_block_size", 2)
residual_projection = genome.get("residual_projection", "auto")
```

Add config values:

```python
"residual_enabled_weight": 0.35,
"residual_disabled_weight": 0.65,
"residual_block_size_options": [2, 3],
"residual_projection_options": ["auto"],
"residual_mutation_weight": 0.15,
"max_model_parameters": None,
```

`max_model_parameters` is optional. When unset, the validator only enforces
shape safety. When set, residual and sequential models whose estimated parameter
count exceeds the limit are rejected before training.

## Model Construction

Introduce small internal modules in `neuroevolution/models/evolvable_cnn.py`:

- `Conv1DUnit`: Conv1D, normalization, activation.
- `ResidualConv1DBlock`: a sequence of `Conv1DUnit` modules, a shortcut path,
  residual addition, final activation, optional dropout, and block-level
  pooling.

For `residual_enabled=False`, keep the existing sequential build path. For
`residual_enabled=True`, group convolutional layers into blocks of
`residual_block_size`. A trailing single layer can fall back to the sequential
unit so odd layer counts remain valid.

The shortcut path uses identity when input and output channels match. It uses
`nn.Conv1d(in_channels, out_channels, kernel_size=1)` when channels differ and
`residual_projection == "auto"`.

Pooling moves to the end of each residual block. This keeps temporal length
unchanged inside the block, making residual addition safe and making deeper
convolutional stacks possible without multiplying downsampling by every conv.

## Validation

Update `is_genome_valid` and `calculate_max_safe_conv_layers` logic so the
effective number of pooling operations is:

```python
if residual_enabled:
    pool_count = ceil(num_conv_layers / residual_block_size)
else:
    pool_count = num_conv_layers
```

The expected temporal length remains `sequence_length / (2 ** pool_count)`.
Validation also normalizes residual fields in `validate_and_fix_genome`.

If parameter estimation is added, implement it as a deterministic helper that
uses genome fields and config rather than instantiating a full model in normal
validation paths.

## Genetic Operators

Genome creation should sample residual fields using config weights and options.
Mutation should be able to:

- Toggle `residual_enabled`.
- Change `residual_block_size`.
- Keep `residual_projection` inside supported options.

Crossover should copy a complete residual topology from one parent or merge
residual genes deterministically. It must rebuild `innovation_genes` after
normalization.

Innovation tracking should include residual genes, for example:

- `residual_enabled`
- `residual_block_size`
- `residual_projection`

Speciation distance should include a small residual topology component so
residual and non-residual lineages are not treated as identical.

## Stable Genome Signature

Any cache signature for equivalent architectures must include the residual
fields:

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
)
```

## Notebook and Reporting

`test.ipynb` should expose the new config keys near the architecture search
configuration and display residual fields in the final best-genome summary.
Reporting utilities should print residual mode, block size, and projection so
saved outputs are interpretable.

## Risks and Mitigations

- **Risk:** Residual blocks increase memory usage.
  **Mitigation:** Keep residual disabled more likely by default, validate
  dimensions before training, and support optional parameter caps.
- **Risk:** Pooling per block changes the meaning of `num_conv_layers`.
  **Mitigation:** Preserve the old path when residual is disabled and document
  the block-level pooling behavior in summaries.
- **Risk:** Crossover can produce partially specified residual fields.
  **Mitigation:** centralize normalization in `validate_and_fix_genome`.

## ResNet Comparison

The implementation keeps the core ideas from the original ResNet paper
(`Deep Residual Learning for Image Recognition`, arXiv:1512.03385):

- stacked convolutional units learn a residual branch;
- the shortcut path is identity when tensor channels match;
- a 1x1 convolution projection is used when channels differ;
- residual branch and shortcut outputs are added before the block output
  activation.

The implementation intentionally adapts those ideas to this project rather
than copying an image ResNet: it uses Conv1D for audio signals, keeps pooling at
the block boundary so temporal length is stable inside the residual addition,
and does not add 2D bottleneck blocks or fixed ResNet depths.
