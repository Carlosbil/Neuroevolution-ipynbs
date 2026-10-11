# Add NEAT-Like Mechanism Tests

## Summary

Add focused unit coverage for the NEAT-like mechanisms used by the neuroevolution package:

- innovation identifiers and structural history,
- innovation-aligned crossover,
- compatibility-based speciation,
- incremental complexity caps.

## Motivation

The article review identified that the method should be characterized carefully as NEAT-like rather than classical NEAT. The code already contains several NEAT-inspired mechanisms, but they are not covered by targeted tests. Unit tests will make those claims auditable and protect future refactors of the genetic operators.

## Scope

In scope:

- Add deterministic unit tests for innovation UUID generation and innovation gene construction.
- Add deterministic unit tests for crossover alignment by `innovation_id`.
- Add unit tests for species assignment, representative updates, and adjusted fitness.
- Add unit tests for incremental complexity cap updates and genome creation under caps.
- Fix small correctness issues exposed by those tests.

Out of scope:

- Implementing classical NEAT node/connection genes.
- Changing the article text.
- Changing the full evolutionary training loop or running expensive model training.
