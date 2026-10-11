# Design

## Test Shape

The tests are pure unit tests under `tests/` and avoid data loading or neural network training. They use small synthetic genome dictionaries with explicit `filters`, `kernel_sizes`, `fc_nodes`, fitness values, and innovation genes.

## Coverage Targets

### Innovation

Verify that innovation UUIDs are deterministic, independent of payload key order, and attached to generated innovation genes and structural history events.

### Crossover

Force crossover randomness to deterministic values using `monkeypatch`. This lets the test verify that homologous genes are aligned by `innovation_id`, disjoint genes can be inherited, offspring are re-identified, and crossover events are written to `structural_history`.

### Speciation

Verify that close genomes are assigned to the same species and distant genomes to separate species. Also verify species representative updates and adjusted fitness calculation.

### Incremental Complexity

Verify that the engine advances current complexity caps by generation stage and that random genome creation respects active caps.

## Implementation Notes

If species assignment fails to group new genomes when no previous species dictionary exists, update `assign_species` so it compares each genome against species created earlier in the same assignment pass.
