# Evolution Selection Specification

## ADDED Requirements

### Requirement: Selection can use multiobjective Pareto ranking

The system SHALL support a selection strategy that ranks individuals by Pareto
dominance over configured objectives.

#### Scenario: Non-dominated individuals form the first front

- **GIVEN** a population with fitness, evaluation time, and parameter count
  objectives
- **WHEN** Pareto ranking is computed
- **THEN** individuals not dominated by any other individual receive
  `pareto_rank = 0`
- **AND** dominated individuals receive lower-priority ranks.

#### Scenario: Fitness is maximized and costs are minimized

- **GIVEN** objectives configured as maximize fitness, minimize evaluation time,
  and minimize parameter count
- **WHEN** dominance is evaluated between two individuals
- **THEN** higher fitness is better
- **AND** lower evaluation time is better
- **AND** lower parameter count is better.

#### Scenario: Tiny fitness differences can be treated as equivalent

- **GIVEN** `pareto_fitness_epsilon` is greater than zero
- **WHEN** two individuals differ in fitness by less than that epsilon
- **THEN** that fitness difference does not by itself dominate the cheaper
  candidate.

### Requirement: Cost objectives are recorded during evaluation

The system SHALL record the cost metadata needed for Pareto selection.

#### Scenario: Evaluation time is stored

- **GIVEN** an individual has completed fitness evaluation
- **WHEN** generation metrics are saved
- **THEN** the individual includes `evaluation_time_seconds`
- **AND** the value is available in logs or progress JSON.

#### Scenario: Parameter count is stored

- **GIVEN** an evaluated genome can be parameter-estimated
- **WHEN** generation metrics are saved
- **THEN** the individual includes `parameter_count`
- **AND** Pareto ranking can use that value as a minimization objective.

### Requirement: Pareto strategy affects elitism and tournament selection

The system SHALL use Pareto metadata for reproduction decisions when Pareto
selection is enabled.

#### Scenario: Pareto elites prefer better fronts

- **GIVEN** `selection_strategy` is `pareto`
- **WHEN** elites are selected for the next generation
- **THEN** individuals from lower Pareto ranks are preferred
- **AND** ties inside a front use diversity-preserving tie-breakers such as
  crowding distance.

#### Scenario: Pareto tournament prefers non-dominated tradeoffs

- **GIVEN** a tournament sample contains a dominated expensive individual and a
  non-dominated cheaper individual with similar fitness
- **WHEN** tournament selection chooses a parent
- **THEN** the non-dominated cheaper individual is preferred.

#### Scenario: Fitness-only strategy remains available

- **GIVEN** `selection_strategy` is `fitness`
- **WHEN** elites or tournament parents are selected
- **THEN** selection uses scalar fitness as before
- **AND** Pareto metadata does not change parent choice.

### Requirement: Pareto selection is auditable

The system SHALL expose objective values and Pareto decisions in artifacts.

#### Scenario: Generation logs include Pareto metadata

- **GIVEN** Pareto selection is enabled
- **WHEN** a generation summary is written
- **THEN** the log includes each individual's Pareto rank
- **AND** includes evaluation time and parameter count.

#### Scenario: Progress JSON preserves objective metadata

- **GIVEN** evolution progress is saved
- **WHEN** `evolution_progress.json` is written
- **THEN** each evaluated individual keeps its objective values
- **AND** saved progress can be resumed without losing Pareto metadata.
