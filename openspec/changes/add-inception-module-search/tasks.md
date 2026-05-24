# Tasks: Add Inception Module Search

## 1. Genome and Config

- [x] 1.1 Add Inception search defaults to `neuroevolution/config.py`.
- [x] 1.2 Add topology sampling for sequential, residual, and Inception modes.
- [x] 1.3 Add Inception fields to random genome creation with
  backward-compatible defaults.
- [x] 1.4 Normalize Inception fields and mutual exclusivity in
  `validate_and_fix_genome`.
- [x] 1.5 Update genome validity checks and parameter estimation for Inception
  branch modules.

## 2. Model Architecture

- [x] 2.1 Add reusable Inception Conv1D branch/channel-splitting helpers.
- [x] 2.2 Add `InceptionConv1DModule` with 1x1, reduced medium, reduced wide,
  and optional pooling-projection branches.
- [x] 2.3 Build Inception conv stacks when `inception_enabled=True`.
- [x] 2.4 Keep existing sequential and residual conv stack behavior unchanged.
- [x] 2.5 Verify output-size calculation works for sequential, residual, and
  Inception stacks.

## 3. Genetic Operators

- [x] 3.1 Add Inception topology mutation with structural history events.
- [x] 3.2 Include Inception genes in innovation tracking.
- [x] 3.3 Update crossover to produce complete, mutually exclusive topology in
  children.
- [x] 3.4 Include Inception topology in speciation distance.
- [x] 3.5 Include Inception topology in any stable genome signature used for
  evaluation cache.

## 4. Notebook, Summaries, and Artifacts

- [x] 4.1 Update `test.ipynb` config/setup cells to expose Inception search
  options.
- [x] 4.2 Update model architecture summaries to show active topology and
  Inception parameters.
- [x] 4.3 Update report utilities and final notebook output for Inception
  architectures.
- [x] 4.4 Ensure checkpoints and progress JSON keep Inception fields without
  special migration code.

## 5. Compare with GoogLeNet

- [x] 5.1 Compare the implemented Inception Conv1D design with GoogLeNet
  Inception v1 to ensure the key branch, reduction, pooling, and concatenation
  features are captured.

## 6. Tests and Verification

- [x] 6.1 Add tests for validating legacy genomes without Inception fields.
- [x] 6.2 Add tests for topology normalization and mutual exclusion with
  residual mode.
- [x] 6.3 Add tests for Inception branch channel splitting and invalid small
  filter counts.
- [x] 6.4 Add tests for Inception model forward pass with and without the pool
  branch.
- [x] 6.5 Add tests for mutation and crossover preserving valid Inception
  topology.
- [x] 6.6 Run the existing test suite plus a lightweight smoke test that builds
  an Inception genome and performs one forward pass.
