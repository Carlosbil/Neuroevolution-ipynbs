# Tasks: Add Residual Block Search

## 1. Genome and Config

- [x] 1.1 Add residual search defaults to `neuroevolution/config.py`.
- [x] 1.2 Add residual fields to random genome creation with backward-compatible defaults.
- [x] 1.3 Normalize residual fields in `validate_and_fix_genome`.
- [x] 1.4 Update genome validity checks to estimate pooling operations differently for residual mode.

## 2. Model Architecture

- [x] 2.1 Add reusable Conv1D unit and residual Conv1D block modules.
- [x] 2.2 Build residual conv stacks when `residual_enabled=True`.
- [x] 2.3 Keep the existing sequential conv stack behavior when residual mode is disabled.
- [x] 2.4 Ensure channel mismatches use automatic 1x1 Conv1D projection.
- [x] 2.5 Verify output-size calculation works for residual and sequential stacks.

## 3. Genetic Operators

- [x] 3.1 Add residual topology mutation with structural history events.
- [x] 3.2 Include residual genes in innovation tracking.
- [x] 3.3 Update crossover to produce complete residual topology in children.
- [x] 3.4 Include residual topology in speciation distance.
- [x] 3.5 Include residual topology in any stable genome signature used for evaluation cache.

## 4. Notebook, Summaries, and Artifacts

- [x] 4.1 Update `test.ipynb` config/setup cells to expose residual search options.
- [x] 4.2 Update model architecture summaries to show residual mode, block size, and projection.
- [x] 4.3 Update report utilities and final notebook output for residual architectures.
- [x] 4.4 Ensure checkpoints and progress JSON keep residual fields without special migration code.

## 5. Compare with ResNet
- [x] 5.1 Compare the implemented residual block design with the original ResNet paper to ensure key features are captured.

## 6. Tests and Verification

- [x] 6.1 Add tests for validating legacy genomes without residual fields.
- [x] 6.2 Add tests for residual genome validation and safe-depth estimation.
- [x] 6.3 Add tests for residual model forward pass with matching and projected shortcuts.
- [x] 6.4 Add tests for mutation and crossover preserving valid residual topology.
- [x] 6.5 Run the existing test suite plus a lightweight smoke test that builds a residual genome and performs one forward pass.
