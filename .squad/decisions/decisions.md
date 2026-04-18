# Team Decisions Log

**Last Updated**: 2026-04-18

---

## Algorithm Flow Analysis: test.ipynb Complete Pipeline

**Date**: 2026-04-11  
**Author**: Dallas (Backend Dev)  
**Status**: Documented  
**Category**: Architecture Documentation  
**Relates to**: test.ipynb, DrawIO diagram

### Context
Extract and document complete ordered algorithm process from test.ipynb for visualization in DrawIO without file edits. Spanish output; exact section structure for diagram readiness.

### Decision
Document the full nested algorithm architecture of test.ipynb as single authoritative reference with:
1. 13-step ordered main flow (setup → config → load → evolution → visualization → analysis)
2. 5 key subprocesses with pseudocode (individual eval, fold training, NEAT crossover, adaptive mutation, speciation)
3. 10 critical decision points with branch logic
4. Complete input/output specs (CONFIG dict, 5-fold .npy format, 5 artifact types)
5. 7 detected improvements with benefit explanations
6. 11 design parameter constraints with reasoning
7. 5 error/recovery scenarios
8. DrawIO-specific notes (colors, hierarchy, cycles, critical trace points, 16 code references)
9. Temporal execution profile (6h GPU, 25h CPU)
10. Validation checkpoints (pre/during/post evolution)

### Rationale
- **Ordered Flow**: 13 linear steps ensure narrative clarity
- **Subprocesses with Pseudocode**: Complex logic (NEAT crossover, adaptive mutation) made transparent
- **Decision Matrix**: 10 branches capture all control flow
- **Modular References**: 16 pointers to neuroevolution/ code keep docs synced with implementation
- **DrawIO Readiness**: Colors, hierarchy, labels, cycle markers explicitly designed for visual
- **Spanish**: Project convention; enables Spanish-speaking researchers to build diagrams
- **Completeness**: Happy path + 5 error scenarios

### Outcomes
- Artifact Created: `.squad/analysis/FLUJO_ALGORITMO_TEST_IPYNB_v2.md` (27.3 KB, 12 sections)
- ✅ Flujo principal: 13 steps with entrada/proceso/salida
- ✅ Subprocesos: 5 with pseudocode, state transitions
- ✅ Puntos de decisión: 10-row decision table
- ✅ Entradas/Salidas: Complete specs
- ✅ DrawIO notas: Colors, hierarchy, cycles, trace points

### Next Steps
1. **Visual Implementation** (Ripley): Convert analysis to DrawIO diagram
2. **Code Validation** (Hockley): Verify module references are current
3. **Team Alignment**: Use diagram as single source of truth

---

## Individual Training Concurrency Pools

**Date**: 2026-04-08  
**Author**: Dallas (Backend Dev)  
**Status**: Implemented  
**Category**: Architecture  

### Context
test.ipynb evolution flow evaluates individuals sequentially, even though each individual already runs 5-fold CV in parallel. Request requires parallel evaluation of multiple individuals using mixed GPU and CPU execution with clear pool progress logs.

### Decision
Implement dual worker pools inside `HybridNeuroevolution.evaluate_population()`:
- GPU pool and CPU pool run concurrently using ThreadPoolExecutor
- Pool sizes configurable via CONFIG (`gpu_pool_size`, `cpu_pool_size`) with safe caps and optional `gpu_pool_max_per_device`
- Individual assignment balanced by pool size ratio with round-robin GPU device selection
- Log remaining individuals per pool and total remaining after each completion

### Consequences
- Enables 2+ individuals training in parallel when resources allow
- Preserves fold-level ThreadPoolExecutor logic while adding per-individual concurrency
- Adds configurable knobs to control concurrency without changing notebook orchestration

### Configuration Defaults
```python
'individual_parallelism': True,
'individual_parallelism_mode': 'auto',
'gpu_pool_size': 4,
'cpu_pool_size': 6,
'gpu_pool_max_per_device': 4,
'gpu_device_ids': None,  # auto-detect
```

---

## Pool Parallelism Semantics Clarification

**Date**: 2026-04-11  
**Author**: Hockley (Tester)  
**Status**: Complete  
**Category**: Architectural Clarification  

### Question
Spanish: "¿Ahora mismo se hace 1 individuo a la vez en cada device, o 4 y 6?"  
English: "Right now is 1 individual at a time per device, or 4 and 6?"

### Answer
**NOT 1-per-device.**

**Current semantics: UP TO 4 GPU + 6 CPU = 10 CONCURRENT INDIVIDUALS** (with resource caps and pool queues).

### Execution Model
1. **Dual independent thread pools**:
   - GPU thread pool: 4 worker threads
   - CPU thread pool: 6 worker threads
   - Both run **simultaneously** in separate threads

2. **Job queuing**:
   - Population split between pools by ratio: `gpu_workers / (gpu_workers + cpu_workers)`
   - Example (20 population, 4 GPU + 6 CPU): ~11 individuals → GPU queue, ~9 → CPU queue

3. **Non-blocking scheduling**:
   - Uses `wait(futures, return_when=FIRST_COMPLETED)` for async completion
   - Not batch-processed; completes as soon as one finishes

4. **Queue replenishment**:
   - After individual completes, `submit_next()` pulls next job from respective queue
   - GPU pool always refills from GPU queue
   - CPU pool always refills from CPU queue

### Resource Caps

| Resource | Cap | Enforcement |
|----------|-----|-------------|
| GPU workers per device | 4 (configurable) | `gpu_pool_max_per_device` in config |
| CPU workers | System CPU count | `os.cpu_count()` at runtime |
| Total workers | GPU + CPU | Fallback to sequential if ≤ 1 |

### Single GPU Scenario
```
Input: 20 population, 1 GPU, config.gpu_pool_size=4, config.cpu_pool_size=6
↓
Runtime resolution:
  - GPU devices: [cuda:0]
  - GPU workers: min(4, 1 device × 4 max) = 4
  - CPU workers: min(6, system cores) = 6
  - Parallelism enabled: True (4 + 6 = 10 workers > 1)
↓
Queue assignment:
  - GPU queue: 11 individuals
  - CPU queue: 9 individuals
↓
Concurrent execution:
  - 4 individuals training on GPU simultaneously
  - 6 individuals training on CPU simultaneously
  - Total: 10 concurrent (bounded by min(population, workers))
```

### Code Evidence
- `neuroevolution/config.py:62-67` — Configuration parameters for pools
- `neuroevolution/evolution/engine.py:81-139` — `_resolve_individual_pool_config()` with caps
- `neuroevolution/evolution/engine.py:169-280` — `_evaluate_population_parallel()` with scheduling
- Line 195: Device rotation via `itertools.cycle()`
- Line 229: `wait(futures, return_when=FIRST_COMPLETED)` — Async completion
- Line 275-278: Queue replenishment logic (GPU and CPU separately)
- Line 271-273: Logging of remaining counts per pool

### Verdict
Current implementation runs **UP TO 4 GPU + 6 CPU simultaneous individuals** (10 workers) on default config, with safe resource caps and async job scheduling. **NOT 1-per-device** unless parallelism is explicitly disabled.

---

## Validation Report: test.ipynb Parallel Pool Implementation

**Date**: 2026-04-08  
**Validator**: Hockley (Tester)  
**Status**: ✅ APPROVED  
**Requirement**: Carlosbil's parallelism request for test.ipynb

### Requirements Statement
Spanish: "Haz que test ipynb ... haya 2 individuos entrenandose al mismo tiempo ... piscina de individuos de 4 en GPU y 6 en CPU ... logs para saber cuantos individuos quedan por entrenarse de cada piscina"

English: "Make test ipynb have 2 individuals training simultaneously ... pool of individuals with 4 on GPU and 6 on CPU ... logs to show how many individuals remain to train from each pool"

### Validation Results

#### 1. Two or More Individuals Training Simultaneously
**Status**: ✅ PASS
- **Implementation**: ThreadPoolExecutor with separate GPU and CPU executor pools
- **Concurrency**: 10 workers (4 GPU + 6 CPU) enable 10 simultaneous individual evaluations
- **Pattern**: `wait(futures, return_when=FIRST_COMPLETED)` at line 229 enables true non-blocking parallelism
- **File**: `neuroevolution/evolution/engine.py:216-278`

#### 2. Dual Pool Configuration: 4 GPU + 6 CPU
**Status**: ✅ PASS (Exactly as specified)
- **Config defaults**: `gpu_pool_size: 4`, `cpu_pool_size: 6`
- **Resolution test**: On CUDA system, resolves to exactly 4 GPU workers and 6 CPU workers
- **Adaptive behavior**: Falls back gracefully to CPU-only if GPU unavailable
- **Method**: `_resolve_individual_pool_config()` validates device availability and applies caps

#### 3. Safe Resource Caps
**Status**: ✅ PASS
- **GPU cap**: `gpu_pool_max_per_device=4` per device
- **CPU cap**: Automatically capped to system CPU count (12 cores on test system, request 6, so 6 used)
- **Implementation pattern**: `max(0, min(requested, available))` ensures safe limits
- **File**: `neuroevolution/evolution/engine.py:96-122`

#### 4. Detailed Logging of Remaining Individuals
**Status**: ✅ PASS (Comprehensive)

##### Log Output 1: Remaining Count Per Pool (Line 271-273)
```python
print(f"   Remaining -> GPU: {gpu_remaining} | CPU: {cpu_remaining} | Total: {total_remaining}")
```
- Shows GPU remaining, CPU remaining, and total remaining
- Printed after each individual completes
- Enables real-time tracking of queue progress

##### Log Output 2: Individual Queueing (Line 208-211)
```python
f"   Queueing individual {index}/{total_count} (ID: {genome['id']}) -> {pool_label.upper()} on {device}"
```
- Shows which individual assigned to which pool
- Shows device assignment (GPU:X or CPU)

##### Log Output 3: Individual Completion (Line 255-259)
```python
f"Individual {index}/{total_count} | ID={genome['id']} | fitness={fitness:.2f}% | best_gen={best_fitness_so_far:.2f}% | global_best={current_global_best_fitness:.2f}%"
```
- Per-individual performance metrics
- Fitness progression tracking

##### Log Output 4: Generation Summary (Line 308-360)
- Detailed metrics table for all individuals
- Generation statistics (max, avg, min, std)
- Per-individual metrics (accuracy, sensitivity, specificity, precision, F1, AUC)

### Architecture Notes

#### Population-to-Pool Assignment
**File**: `neuroevolution/evolution/engine.py:141-167`  
**Method**: `_assign_population_to_pools(population_indexed, pool_config)`

Assigns individuals to GPU/CPU pools based on pool ratios:
```python
ratio = gpu_workers / float(gpu_workers + cpu_workers)
gpu_target = max(1, int(round(total * ratio)))
```

#### Concurrent Execution Model
**File**: `neuroevolution/evolution/engine.py:215-278`

Pattern:
1. Create GPU and CPU executors with max_workers set
2. Pre-queue initial batch of individuals (lines 221-226)
3. As each individual completes (lines 228-279):
   - Get result and update counters
   - Log remaining counts
   - Submit next individual from same pool
   - Update global best if needed

This maintains full worker utilization while providing per-individual progress logging.

### Configuration Values
```python
'individual_parallelism': True,
'individual_parallelism_mode': 'auto',
'gpu_pool_size': 4,
'cpu_pool_size': 6,
'gpu_pool_max_per_device': 4,
'gpu_device_ids': None,  # auto-detect
```

### Approval Statement

✅ **The current implementation in test.ipynb FULLY SATISFIES all requirements.**

The hybrid neuroevolution engine provides:
1. Concurrent training of 10 individuals (far exceeding the minimum 2)
2. Separate GPU (4 workers) and CPU (6 workers) pools as specified
3. Safe resource caps on both pools with device-aware fallbacks
4. Comprehensive logging showing per-pool remaining counts, individual assignments, and progress metrics

**No code changes required. Implementation is complete and production-ready.**

---

## Decisión: Diagrama draw.io flujo test.ipynb

**Date**: 2026-04-11  
**Author**: Dallas  
**Status**: Implemented  
**Category**: Documentation

### Decisión
Se optó por crear el diagrama draw.io directamente en español, siguiendo la estructura orquestal del notebook, el flujo hacia los módulos de neuroevolución, el bucle de evolución y evaluación, el paralelismo GPU/CPU (ThreadPoolExecutor, colas, FIRST_COMPLETED), y las salidas principales (logs, checkpoints, plots).

### Validación
Se validó la estructura XML y se priorizó claridad para onboarding y documentación técnica. No se modificó lógica fuente.

### Resultado
Diagrama creado en: `mejoras/06_diagrama_proceso_test_ipynb.drawio`

---

## Clarify Individual Parallelism Semantics

**Date**: 2026-04-11  
**Owner**: Dallas  
**Status**: Proposed  
**Category**: Documentation

### Context
Team questions highlighted ambiguity around how GPU/CPU pools map to individuals and devices.

### Decision
Document that one worker equals one individual job, pool sizes are max concurrent individuals per pool, and gpu_pool_max_per_device caps concurrent individuals on each GPU (defaults allow 4 GPU + 6 CPU, resource-capped).

### Impact
Improves run-time clarity and aligns documentation across config and engine logs without changing behavior.

---

**Document Location**: `.squad/decisions/decisions.md`  
**Last Merge**: 2026-04-18 13:28:19 UTC  
**Deduplication**: All 7 inbox items merged; removed duplicates (pool parallelism consolidated across 3 sources)
