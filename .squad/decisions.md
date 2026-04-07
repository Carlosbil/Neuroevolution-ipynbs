# .squad/decisions.md

## Active Decisions

### 1. Module Architecture for test_simplified.ipynb Refactoring
**Date**: 2025-01-22  
**Lead**: Ripley  
**Status**: Approved  

19-module architecture across 7 packages with 7-phase roadmap. Preserves Jupyter-centric research workflow while enabling modularity and testability.

**Key Decisions**:
- NEAT-like innovation UUID tracking for gene alignment
- ThreadPoolExecutor for parallel 5-fold CV (not Process pool)
- Three-tier validation strategy (prevention, fixing, runtime)
- Single source of truth for config management
- Checkpoint format compatibility (.pth with metadata)

**Modules**: 7 packages, 19 modules, ~78 KB total code  
**Phases**: 1-7 from foundation to visualization

---

### 2. Phase 1-3 Implementation Complete
**Date**: 2025-01-22  
**Author**: Dallas  
**Status**: Completed  

Successfully extracted and modularized foundational components (Phases 1-3) from `test_simplified.ipynb` into the `neuroevolution` package. All code preserves exact logic with ZERO algorithm changes.

**Modules Delivered**:
- **Phase 1**: config.py, logger.py, device_utils.py
- **Phase 2**: data/loader.py, models/evolvable_cnn.py, models/genome_validator.py
- **Phase 3**: genetics/{innovation.py, genome.py, mutation.py, crossover.py, selection.py, speciation.py}

**Code Quality**: 16 files, ~58 KB, validation-first approach, OS-independent paths  
**Next**: Phases 4-7 (evolution engine, evaluation, visualization)

---

### 3. Testing Infrastructure Setup Complete
**Date**: 2025-01-22  
**Lead**: Hockley  
**Status**: Ready for module implementation  

Comprehensive validation strategy ensuring exact numerical equivalence between refactored modules and original notebook. Zero-tolerance philosophy with 32 test suites across 4 levels.

**Deliverables**:
- Validation strategy document (41 KB) with complete test plan
- Pytest infrastructure: conftest.py, utils.py, pytest.ini
- Comparison utilities for genomes, tensors, models, data, evolution progress
- Artifact generation script for ground truth baselines
- Example test suite (7/7 tests passing on CUDA)

**Test Levels**: 19 unit + 3 integration + 6 regression + 4 performance test suites  
**Acceptance Criteria**: 100% pass rate, strict tolerances (exact for integers, ±1e-7 for float32, ±1e-5 for CUDA)  
**Status**: Infrastructure production-ready, awaiting module implementations

---

### 4. Validation Strategy: Multi-Level Testing Pyramid
**Date**: 2025-01-22  
**Author**: Hockley  
**Status**: Approved  

4-level testing approach for exact numerical equivalence validation:
1. **Unit tests** (19 modules): Isolated validation with mocked dependencies
2. **Integration tests** (3 suites): Multi-module interaction validation
3. **Regression tests** (6 suites): Byte-for-byte comparison with notebook outputs
4. **Performance tests** (4 suites): Numerical precision and parallelism validation

**Floating-Point Tolerance Strategy**:
- Integer operations: Exact match (0 tolerance)
- Float32: ±1e-7 (PyTorch default precision)
- Float64: ±1e-15 (NumPy double precision)
- CUDA: ±1e-5 (hardware non-determinism)
- Accumulated errors: ±1e-5 for long training runs

**Critical Artifacts**: fold_0_data.npz, 100_random_genomes.json, 1000_mutations.json, 100_crossover_children.json, mini_evolution_progress.json, final_cv_results.json

**Risk Mitigation**: Comprehensive strategies for parallel CV, NEAT crossover complexity, CUDA non-determinism, checkpoint compatibility

---

## Archived Decisions

_Decisions older than 30 days will be moved here._
