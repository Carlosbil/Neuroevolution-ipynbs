---
name: "individual-parallelism-pools"
description: "Run per-individual evolution evaluations concurrently with GPU/CPU pools"
domain: "performance"
confidence: "medium"
source: "earned"
---

## Context
Applies when evaluating many genomes sequentially is slow, but each genome already uses fold-level parallelism. Use dual GPU/CPU pools to execute multiple individuals concurrently while keeping notebook orchestration unchanged.

## Patterns
- Add CONFIG knobs for pool sizes and modes (`individual_parallelism`, `gpu_pool_size`, `cpu_pool_size`, `gpu_pool_max_per_device`, `individual_parallelism_mode`).
- Resolve pool sizes with safe caps (CPU cores, GPU availability) and fallback to sequential when only one worker is possible.
- Maintain two ThreadPoolExecutors (GPU + CPU) and submit tasks from separate queues.
- Log per-pool remaining counts after each completion to track progress.
- Document worker semantics (one worker == one individual job) and per-pool/per-device caps in config and engine docstrings/logs.

## Examples
- `neuroevolution/evolution/engine.py`: `_resolve_individual_pool_config()` and `_evaluate_population_parallel()` implement dual pools and progress logs.

## Anti-Patterns
- Using ProcessPoolExecutor for GPU workloads (breaks CUDA context).
- Overwriting algorithm logic or hyperparameters when adding concurrency.
- Running without logging per-pool progress (hard to audit queue depth).
