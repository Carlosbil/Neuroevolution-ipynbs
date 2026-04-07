# Skill: Notebook-to-Modules Refactoring for Research Code

**Pattern**: Extract inline notebook code into modular Python packages while preserving exact behavior equivalence.

**Context**: Large Jupyter notebooks (3,000+ lines) with scientific algorithms need to be refactored for modularity, testing, and reusability without losing reproducibility.

**Challenge**: Research code often depends on implicit state (seeds, device, config), side effects (print redirection), and precise numerical operations that break if moved incorrectly.

---

## The Pattern

### 1. **Map Before You Code**

Before touching any code, produce a comprehensive mapping:

| Phase | Notebook Cells | Lines | Content | Target Module |
|-------|---|---|---|---|
| 1 | 1-3 | 283 | Setup, imports, logging | (orchestrator) |
| 2 | 2 | 105 | Configuration | (orchestrator) |
| 3 | 3 | 175 | Data loading | data/loader.py |
| ... | ... | ... | ... | ... |

**Why**: Prevents mid-refactoring discovery that you've split logic across cells inconsistently.

### 2. **Identify Critical Invariants**

These CANNOT change:
- Seed initialization timing and values
- Random number stream sequence
- Floating-point precision in calculations
- File paths and fallback logic
- Configuration dict completeness
- Checkpoint format and metadata

**Why**: One change breaks reproducibility for ALL downstream runs.

### 3. **Implement in Phases**

Never refactor all at once. Instead:

1. **Prep phase**: Extract helper functions (pure logic, no side effects)
2. **Config phase**: Centralize all global state
3. **Core phase**: Move main algorithms
4. **Integration phase**: Assemble into classes/modules
5. **Test phase**: Verify equivalence with original
6. **Refactor phase**: Clean up, add types, documentation
7. **Orchestration phase**: Make notebook thin

**Why**: Each phase has clear entry/exit criteria for testing.

### 4. **Preserve State Order**

In Jupyter, implicit state matters:

```python
# BAD: Extract this in any order
seed_set() → device_set() → logger_setup() → config_load() → data_load()
# Breaks if seed_set() happens after data_load()

# GOOD: Document strict order
1. Set seed (FIRST)
2. Configure device
3. Setup logger (may use random for some reason)
4. Load config
5. Load data
```

**Why**: Random streams diverge if seeds are set after first random call.

### 5. **Test Each Phase**

Compare notebook vs. module at every phase:

```python
# Phase 1 (setup): Verify seeds produce same random values
notebook_random_val = ... # from notebook cell 1
module_random_val = from_module()
assert notebook_random_val == module_random_val

# Phase 2 (config): Verify CONFIG dict identical
notebook_config = ... # from notebook cell 2
module_config = get_default_config()
assert notebook_config == module_config

# Phase 3 (data): Verify data loaded from same location
notebook_data = ... # from notebook cell 3
module_data = load_dataset(config)
assert np.array_equal(notebook_data, module_data)

# ... etc for each phase
```

**Why**: Catches equivalence breaks early when they're easy to fix.

### 6. **Document Non-Obvious Logic**

For each extracted function, document:
- **Preconditions**: What state must exist before calling
- **Side effects**: What global state is modified
- **Invariants**: What must remain constant
- **Equivalence notes**: Why this differs from naive refactoring

Example:
```python
def mutate_genome(genome: dict, config: dict) -> dict:
    """
    Mutates a genome while preserving innovation history.
    
    INVARIANTS (DO NOT CHANGE):
    - Mutation rate must come from config['current_mutation_rate'], not hardcoded
    - Random operations must preserve seed sequence (use global RNG, not new generators)
    - Innovation UUID must use INNOVATION_NAMESPACE constant from innovation.py
    - Structural events must append to structural_history, not replace
    - Output must be validated with validate_and_fix_genome() before returning
    
    EQUIVALENCE NOTE:
    - This function is called by tournament_selection() which expects exact same
      genome mutations as notebook cell 5-6. Do NOT optimize or parallelize.
    
    PRECONDITIONS:
    - config['current_mutation_rate'] must be set (use in-loop, not global)
    - genome must pass is_genome_valid() before mutation
    - Seed must be set before calling (inherited from main loop)
    """
```

**Why**: Prevents "helpful" refactoring that breaks subtle invariants.

---

## Anti-Patterns to Avoid

### ❌ "Let me optimize this while I'm here"

```python
# NOTEBOOK CODE (inefficient but correct)
for i in range(len(genome['filters'])):
    if random.random() < mutation_rate:
        genome['filters'][i] = random.randint(min_val, max_val)

# TEMPTING "OPTIMIZATION" (wrong!)
genome['filters'] = [
    random.randint(min_val, max_val) if random.random() < mutation_rate else f
    for i, f in enumerate(genome['filters'])
]
# WHY WRONG: Changes random number sequence (two random() calls per element)
```

**Fix**: Copy exact logic as-is, even if ugly.

### ❌ "This formula can be simplified"

```python
# NOTEBOOK
new_rate = config['base_mutation_rate'] + (inverted - 0.5) * 0.4
new_rate = max(config['mutation_rate_min'], min(config['mutation_rate_max'], new_rate))

# TEMPTING "SIMPLIFICATION"
new_rate = np.clip(config['base_mutation_rate'] + (inverted - 0.5) * 0.4, 
                   config['mutation_rate_min'], 
                   config['mutation_rate_max'])
# WHY WRONG: Different floating-point rounding order can cause 1e-8 differences
```

**Fix**: Use exact same operations in exact same order.

### ❌ "Config should be immutable"

```python
# TEMPTING REFACTORING
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    population_size: int
    max_generations: int
    ...

# WHY WRONG: Notebook modifies config dynamically:
# config['current_mutation_rate'] = new_rate  # Set each generation
# config['current_max_conv_layers'] = min(max_conv, initial + stage)  # Updated
```

**Fix**: Keep CONFIG as mutable dict, document what's mutable vs. read-only.

### ❌ "Let me refactor checkpoint format"

```python
# NOTEBOOK CHECKPOINT FORMAT (verbose but must preserve)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'genome': genome,
    'generation': self.generation,
    'fitness': genome['fitness'],
    'config': self.config
}

# TEMPTING "CLEANER" FORMAT
checkpoint = {
    'model': model,  # DON'T: Can't pickle model directly
    'genome': genome,
    'metadata': {'gen': ..., 'fitness': ...}  # DON'T: Changed keys
}
```

**Fix**: Keep checkpoint format exactly as-is.

### ❌ "Parallelizing will make it faster"

```python
# NOTEBOOK: Sequential 5-fold CV with ThreadPoolExecutor for parallel folds
# But within each fold, training is sequential

# TEMPTING PARALLEL: "Let me parallelize training within fold too"
# with ThreadPoolExecutor(max_workers=4):  # Parallelize batches
#     ...

# WHY WRONG: Changes random order, gradient updates order, CUDA timing
```

**Fix**: Keep parallelization structure exactly as notebook.

---

## Equivalence Testing Checklist

After refactoring each phase, verify:

- [ ] **Seed equivalence**: Same SEED values produce identical random sequences
- [ ] **Data equivalence**: Data loaded from same paths, same shapes, same values
- [ ] **Numeric equivalence**: Float calculations differ by <1e-7 (float32) or <1e-15 (float64)
- [ ] **Structure equivalence**: Dicts/lists have identical keys/lengths
- [ ] **State equivalence**: Global state (device, logger) is identical
- [ ] **Output equivalence**: Printed output matches (can be tested with string comparison)
- [ ] **Behavior equivalence**: Full evolution produces same generation stats (within tolerance)

**Test methodology**:

```python
def test_equivalence_phase_N(notebook_output, module_output):
    """Compare notebook cell N output with module implementation."""
    
    if isinstance(notebook_output, np.ndarray):
        np.testing.assert_allclose(
            notebook_output, module_output,
            rtol=1e-5,  # Relative tolerance for float
            atol=1e-7   # Absolute tolerance for float
        )
    elif isinstance(notebook_output, dict):
        assert set(notebook_output.keys()) == set(module_output.keys())
        for key in notebook_output:
            test_equivalence(notebook_output[key], module_output[key])
    elif isinstance(notebook_output, list):
        assert len(notebook_output) == len(module_output)
        for i in range(len(notebook_output)):
            test_equivalence(notebook_output[i], module_output[i])
    elif isinstance(notebook_output, float):
        assert abs(notebook_output - module_output) < 1e-7
    else:
        assert notebook_output == module_output
```

---

## Quick Checklist for Implementation

### Before Starting
- [ ] Read notebook 3-4 times, understand flow
- [ ] Map cells to modules (create table)
- [ ] Identify critical invariants (list them)
- [ ] Create baseline from notebook (generate example outputs)
- [ ] Plan import order (draw dependency graph)

### During Implementation
- [ ] Implement one module at a time
- [ ] Copy logic exactly (even if inefficient-looking)
- [ ] Test each module in isolation
- [ ] Test each module against notebook baseline
- [ ] Document non-obvious logic with EQUIVALENCE NOTES
- [ ] Do NOT refactor, optimize, or parallelize

### After Implementation
- [ ] Run full integration test (end-to-end)
- [ ] Compare all outputs byte-for-byte
- [ ] Document any acceptable tolerance (±1e-5 etc)
- [ ] Test from different working directories (path resolution)
- [ ] Test with different seeds (reproducibility)
- [ ] Mark non-code changes in notebook (markdown, comments)

---

## Common Pitfalls & Fixes

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Seed set too late | Random sequences diverge after cell 1 | Move seed to absolute first operation |
| Config incomplete | AttributeError on missing key | Use `get_default_config()` with all keys |
| Path hardcoded | Fails on different machine | Use `os.path.join()` with fallback resolution |
| Mutation rate hardcoded | Evolution curves don't match | Use `config['current_mutation_rate']` |
| Checkpoint format changed | Can't load old checkpoints | Keep dict keys identical |
| Floating-point rounding | Tiny differences accumulate | Use exact same operations in exact order |
| ThreadPoolExecutor workers changed | Race conditions, different results | Keep `max_workers=5` |
| Device set late | Model on GPU, eval on CPU | Set device before any tensor creation |
| Early stopping threshold changed | Converges faster/slower | Use exact same patience/threshold values |

---

## References

- **Jupyter best practices**: Keep notebook state explicit, document cell dependencies
- **Reproducible research**: Track all sources of non-determinism (seeds, devices, rounding)
- **Testing patterns**: Unit → Integration → Regression (baseline comparison)
- **Refactoring**: Extract method, move method, extract class (in that order)

---

## Applied Pattern Addendum (test_simplified.ipynb)

When converting a long research notebook into an orchestrator notebook, preserve a **variable contract** across cells:

- Setup phase must still define: `info_path`, `logger`, `SEED`, `device`
- Config phase must still define: `CONFIG`, `ACTIVATION_FUNCTIONS`, `OPTIMIZERS`
- Evolution phase must still define: `neuroevolution`, `best_genome`, `execution_time`
- Final evaluation phase must still define: `cv_results`

This lets markdown context and downstream cells remain valid while all implementation moves to modules.
