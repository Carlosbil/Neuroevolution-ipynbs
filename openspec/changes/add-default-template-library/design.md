## Context

`architecture_templates.py` already has the registry, descriptor type, genome
factory, provenance handling, and mutation helper needed for known Conv1D
templates. The current registry only contains `resnet_conv1d_small` and
`googlenet_inception_conv1d_small`.

The cheapest extension is to add more descriptors that map to existing genome
fields. The model builder already supports sequential, residual, and Inception
Conv1D topologies, so no new PyTorch modules are needed.

## Goals / Non-Goals

**Goals:**

- Add a broader default set of valid template descriptors.
- Cover sequential, residual, and Inception topology families.
- Keep every new template compatible with existing validation, mutation,
  crossover, parameter estimation, and model construction.
- Add tests that every default template can build an `EvolvableCNN` and perform
  a forward pass under a small config.

**Non-Goals:**

- Do not add dilated, depthwise-separable, dense-connectivity, or squeeze/expand
  block types.
- Do not change template seeding quotas or mutation probabilities.
- Do not claim exact reproduction of original image architectures; these are
  Conv1D adaptations.

## Decisions

Add these template IDs to the registry and default config:

- `lenet_conv1d_tiny`
- `alexnet_conv1d_small`
- `vgg_conv1d_small`
- `resnet_conv1d_small`
- `resnet_conv1d_medium`
- `wide_resnet_conv1d_small`
- `googlenet_inception_conv1d_small`
- `googlenet_inception_conv1d_medium`
- `inception_conv1d_wide`

Sequential templates use plain Conv1D layer lists with different depth/channel
profiles. Residual templates use `conv_topology="residual"` and existing
`residual_block_size` fields. Inception templates use
`conv_topology="inception"` and existing Inception branch options.

All template channel counts must remain small enough to pass the existing test
configuration while still differentiating families. Runtime configuration can
still clamp filters, kernels, and layer counts through existing factory logic.

## Risks / Trade-offs

- More default templates can reduce random diversity -> existing seed fraction
  and minimum random fraction continue to cap template use.
- Names such as AlexNet/VGG can imply exact image models -> IDs include
  `conv1d` and descriptors remain genome adaptations.
- Template defaults can become stale as search space evolves -> tests should
  verify all configured defaults remain valid and buildable.
