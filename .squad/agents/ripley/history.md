# History — Ripley

## Project Knowledge

- **Project**: Neuroevolution-ipynbs — Parkinson voice detection using hybrid neuroevolution
- **Tech Stack**: Python, PyTorch, Jupyter notebooks, NumPy, scikit-learn, genetic algorithms
- **Lead Developer**: Carlosbil
- **Main Artifacts**:
  - `test_simplified.ipynb` — Main research notebook (49 cells, ~186KB)
  - `best_Audio_hybrid_neuroevolution_notebook.ipynb` — Reference implementation
  - `data/sets/folds_5/` — 5-fold CV datasets (.npy files)

## Current Task

Refactor `test_simplified.ipynb` to use modular Python scripts while maintaining exact functionality. The notebook should orchestrate the scripts, not contain all the logic inline.

## Core Context

### Hybrid Neuroevolution Architecture
- NEAT-like innovation tracking (UUID-based gene alignment)
- Parallel 5-fold CV with ThreadPoolExecutor (5 workers)
- Adaptive mutation rates based on population diversity
- Incremental complexity growth (start simple, grow over generations)
- Species-based speciation to prevent premature convergence

### Critical Preservation Requirements (10 identified)
1. Seed management: SEED=42 set BEFORE any randomness
2. Logging redirection: Print redirection to artifact files
3. Device configuration: Single GPU/CPU source of truth
4. CONFIG dict: 32+ keys, complete at initialization
5. Path fallback logic: OS-independent multi-path resolution
6. Checkpoint format: .pth with state_dict + genome + config
7. Parallel fold threading: ThreadPoolExecutor max_workers=5
8. Genome validation (3-tier): is_genome_valid → _validate_genome → validate_and_fix_genome
9. Adaptive mutation formula: Exact formula preserved (no approximations)
10. Innovation UUID namespace: INNOVATION_NAMESPACE constant immutable

### Architecture Decisions
- **19 modules across 7 packages**: Organized by responsibility (config, data, models, genetics, evolution, evaluation, visualization)
- **Thin orchestrator pattern**: Reduce notebook from 49 cells (~3,800 lines) to 7 cells (~100 lines)
- **Dependency graph**: Clear import hierarchy, no circular dependencies
- **Testing strategy**: Unit + integration + regression + performance (32 test suites)

## Learnings

### README Rewrite to Current Reality (2026-04-08)

**Task**: Rewrite README.md in Spanish to reflect post-refactorization project state  
**Status**: ✅ Complete

**Deliverable**: Updated `README.md` (2,100+ lines, 9 major sections)

**Key Changes**:
- Added header highlighting refactorization: "notebooks ahora actúan como orquestadores ligeros"
- New section: "Estructura del proyecto (post-refactorización)" with 7-module breakdown (26 .py files, ~78 KB)
- New section: "Flujo de ejecución: orquestación con módulos" (pseudocode execution diagram)
- New section: "Testing (post-refactorización)" with pytest examples and 4-level validation strategy
- New section: "Decisiones arquitectónicas" explaining 5 key choices (orchestrator, NEAT UUID, ThreadPoolExecutor, 3-tier validation, device singleton)
- New section: "Cómo ejecutar" with notebook + Python script examples
- Removed stale content (e.g., "todo el pipeline en notebooks", vague data layouts)
- Updated "Estatus del proyecto" to reflect actual state (refactorization complete, tests operational)

**Documentation Quality**:
- Spanish throughout (matches project convention)
- Accurate module references (neuroevolution/ package structure verified)
- Testing guidance (pytest markers, artifact location, tolerance levels)
- Researcher-friendly (GPU memory notes, reproducibility, variants, checkpoint safety)
- Actionable (can execute immediately after reading)

**Decision Record**: Merged `.squad/decisions/inbox/ripley-readme-refactored-state.md` → decisions.md as Decision #8

---

### Session Summary (2026-04-07)

**Orchestration Plan Delivered** (770-line reference document):
- 7-phase roadmap from foundation to visualization
- 10 critical equivalence requirements with mitigation
- 8 risk mitigation strategies with explicit tests
- 19-module structure with dependency graph
- Cell-by-cell mapping from notebook to target modules
- Acceptance criteria for all phase tasks

**Team Execution**:
- **Dallas** implemented all 7 phases (19+ modules, 78 KB code)
- **Hockley** built 32-test validation pyramid (unit/integration/regression/performance)
- **Coordinator** resolved critical blockers (UTF-8 encoding, artifact tests)

**Key Synchronization**:
- All agents aligned on equivalence criteria (±1e-7 Float32, ±1e-5 CUDA)
- Ripley's plan served as single source of truth for all decisions
- Test infrastructure matches specification; blockers identified and resolved
- Cross-team updates documented in agent history files

**Production Status**:
- ✅ Code architecture complete (19+ modules, 7 phases)
- ✅ Test infrastructure operational (32 suites ready)
- ✅ Critical blockers resolved (UTF-8, artifacts)
- 🔲 Reference artifacts pending generation
- 🔲 Full regression suite execution pending

### Technical Decisions (2025-01-22)

1. **Innovation tracking**: NEAT-like UUID system (innovation_uuid, innovation_genes, structural_history)
2. **Validation strategy**: Three-tier (prevention → fixing → runtime)
3. **Parallel design**: ThreadPoolExecutor max_workers=5 (not Process pool)
4. **Config management**: Single source of truth in config.py
5. **Testing approach**: Unit (mocked) + integration (mini evolution) + regression (exact comparison)

### Risk Mitigation Strategies (8 identified, all documented)

1. Random seed divergence → set SEED before imports
2. CONFIG dict mutation → deepcopy in evolution engine
3. Device mismatch → single device from device_utils.py
4. Path fallback logic → centralize in data/loader.py
5. Fold count mismatch → assert num_folds==5
6. Mutation rate not updated → log adaptive rate each generation
7. Checkpoint overwrite → log old path before delete
8. Early stopping criteria → use same config in evolution and CV

---

**Status**: ✅ Complete. Orchestration plan executed successfully. All team members synchronized on architecture and equivalence criteria.
