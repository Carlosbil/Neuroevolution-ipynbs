## Why

The template registry currently proves the mechanism with one ResNet-like and
one GoogLeNet-like Conv1D seed. A broader default library gives evolution more
useful starting points across cheap baselines, deeper sequential stacks, wider
residual variants, and medium Inception variants without adding new model
building primitives.

## What Changes

- Add additional default architecture templates that use only the existing
  sequential, residual, and Inception Conv1D topology modes.
- Include compact sequential baselines inspired by LeNet, AlexNet, and VGG.
- Include medium and wide residual templates inspired by ResNet/WideResNet.
- Include medium and wide Inception templates inspired by GoogLeNet/Inception.
- Make the new templates available through the configured default template ID
  list.
- Keep TCN, MobileNet, DenseNet, SqueezeNet, and other templates requiring new
  block types out of scope for this change.

## Capabilities

### New Capabilities

- `default-template-library`: A broader default set of known architecture
  templates that can be converted into valid searchable Conv1D genomes.

### Modified Capabilities

None.

## Impact

This affects the architecture template registry, default configuration,
notebook configuration, and tests that validate template availability,
validity, and model construction. It does not add dependencies or new PyTorch
module families.
