## 1. Template Registry and Configuration

- [x] 1.1 Add `neuroevolution/genetics/architecture_templates.py` with template descriptors, registry lookup, and genome factory helpers for `resnet_conv1d_small` and `googlenet_inception_conv1d_small`.
- [x] 1.2 Ensure template factories return normalized genomes that pass `validate_and_fix_genome`, `is_genome_valid`, and `build_innovation_genes`.
- [x] 1.3 Add default template configuration keys to `neuroevolution/config.py`.
- [x] 1.4 Extend `validate_config` to reject invalid template fractions, mutation weights, duplicate IDs, and unknown template IDs.

## 2. Initial Population Seeding

- [x] 2.1 Add a helper in `HybridNeuroevolution` to compute the template seed quota while preserving the configured random seed fraction and double-cap slot.
- [x] 2.2 Integrate template-derived individuals into `initialize_population` using the same quartile and incremental cap flow as random genomes.
- [x] 2.3 Record `architecture_template_seed` structural events with template ID, family, origin, quartile, and cap metadata.
- [x] 2.4 Keep existing random quartile initialization and double-cap seeding behavior unchanged when template seeding is disabled.

## 3. Template Module Mutation

- [x] 3.1 Add a bounded template module mutation helper in `neuroevolution/genetics/mutation.py`.
- [x] 3.2 Wire `architecture_template_mutation_weight` into `mutate_genome` without bypassing existing mutation paths.
- [x] 3.3 Ensure template mutation clears stale evaluation cache fields, normalizes the genome, validates safety, rebuilds innovation genes, and assigns a fresh ID.
- [x] 3.4 Record `architecture_template_module` structural events with template ID, family, origin, and affected topology or Conv1D segment metadata.

## 4. Provenance, Formatting, and Artifacts

- [x] 4.1 Normalize optional template provenance fields on template-derived genomes while preserving existing random genome behavior.
- [x] 4.2 Update architecture formatting, `EvolvableCNN.get_architecture_summary`, reports, checkpoint summaries, and progress artifacts to display template provenance when present.
- [x] 4.3 Review `stable_genome_signature` and speciation distance so normalized architecture differences remain cache-safe and provenance does not over-separate evolved descendants.
- [x] 4.4 Update `test.ipynb` configuration cells to expose the new template controls near architecture search settings.

## 5. Tests

- [x] 5.1 Add tests for registry lookup, unknown template validation, and template factory output validity.
- [x] 5.2 Add tests that initial population seeding respects seed fraction, minimum random fraction, quartile caps, and double-cap reservation.
- [x] 5.3 Add tests that template module mutation returns valid genomes, records structural history, and keeps standard mutation available.
- [x] 5.4 Add tests for provenance visibility in summaries, formatting, progress serialization, and checkpoint/report output helpers.
- [x] 5.5 Run the focused pytest suite for architecture templates, genome validation, mutation, crossover, reporting, residual, and Inception behavior.
