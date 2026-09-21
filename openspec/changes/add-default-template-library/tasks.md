## 1. Tests First

- [x] 1.1 Add a failing test that asserts the expanded default template IDs are present in the registry and default config.
- [x] 1.2 Add a failing test that creates every default template genome, validates it, builds `EvolvableCNN`, and runs a small forward pass.

## 2. Registry and Defaults

- [x] 2.1 Add sequential template descriptors for `lenet_conv1d_tiny`, `alexnet_conv1d_small`, and `vgg_conv1d_small`.
- [x] 2.2 Add residual template descriptors for `resnet_conv1d_medium` and `wide_resnet_conv1d_small`.
- [x] 2.3 Add Inception template descriptors for `googlenet_inception_conv1d_medium` and `inception_conv1d_wide`.
- [x] 2.4 Update default template IDs in config so all registry defaults are enabled by default.

## 3. Notebook and Verification

- [x] 3.1 Update `test.ipynb` architecture template list to include the expanded defaults.
- [x] 3.2 Run the focused architecture-template tests.
- [x] 3.3 Run OpenSpec strict validation for this change.
