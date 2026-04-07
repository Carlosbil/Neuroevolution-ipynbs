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

## Learnings

### Architecture Patterns

- **Jupyter-centric research workflow**: The project is built around notebooks for experimentation, traceability, and rapid configuration comparison. The refactoring must preserve this workflow by keeping the notebook as the orchestrator.
  
- **Hybrid neuroevolution structure**: The system combines genetic algorithms (architecture evolution) with supervised training (weight optimization). Key architectural patterns:
  - NEAT-like innovation tracking using UUID-based gene alignment
  - Parallel 5-fold cross-validation with ThreadPoolExecutor (5 workers)
  - Adaptive mutation rates based on population diversity
  - Incremental complexity growth (start simple, grow over generations)
  - Species-based speciation to prevent premature convergence

- **Critical preservation requirements**:
  - Exact CONFIG dictionary structure (32+ parameters)
  - Print() redirection to artifact logs (reproducibility)
  - Genome validation (prevent BatchNorm spatial dimension errors)
  - Multi-path dataset fallback logic (OS-independent)
  - Checkpoint format (.pth with state_dict + genome + config)
  - Seed management (SEED=42 for torch, numpy, random, cuda)

### Module Organization Decisions

- **19 Python modules across 7 packages**: Organized by responsibility (config, data, models, genetics, evolution, evaluation, visualization)
  
- **Thin orchestrator pattern**: Reduce notebook from 49 cells (~3800 lines) to 6-8 cells (~50-100 lines) by extracting all logic to importable modules

- **Dependency graph design**: Keep modules loosely coupled with clear import hierarchy (config at root, no circular deps)

### Key File Paths

- **Main notebook**: `test_simplified.ipynb` (49 cells, ~186KB)
- **Reference implementation**: `best_Audio_hybrid_neuroevolution_notebook.ipynb`
- **Dataset location**: `data/sets/folds_5/files_real_40_1e5_N/`
- **Artifacts output**: `artifacts/test_audio/` (evolution_progress.json, best_model_checkpoint.pth, execution_log.txt)
- **Module root**: `neuroevolution/` (to be created)

### User Preferences (inferred from custom instructions)

- **Reproducibility is paramount**: All output must match exactly (given same seed)
- **Research-grade quality**: LaTeX/Markdown table generation for papers
- **Parallel performance**: ThreadPoolExecutor for 5-fold CV is non-negotiable
- **Validation-heavy**: Multiple genome validation layers to prevent runtime errors
- **Artifact-centric logging**: All print() output goes to files, not just console

### Team Coordination Notes

- **Dallas** (Implementation): Implemented Phases 1-3 (foundation → core components → genetics), proceeding with Phases 4-7
- **Hockley** (Testing): Validation infrastructure complete, ready for parallel test development
- **Carlosbil** (Owner): Project owner, reviews milestones
- **Ripley** (me): Owns architecture decisions, reviews code quality, manages scope

### Technical Decisions

1. **Innovation tracking**: Keep NEAT-like UUID system (innovation_uuid, innovation_genes, structural_history)
2. **Validation strategy**: Three-tier (is_genome_valid, validate_and_fix_genome, EvolvableCNN._validate_genome)
3. **Parallel design**: ThreadPoolExecutor with max_workers=5 for fold training (not Process pool due to CUDA/thread safety)
4. **Config management**: Single source of truth (config.py), runtime updates allowed
5. **Testing approach**: Unit tests (mocked), integration tests (mini evolution), regression tests (exact comparison)

## Refactoring Kickoff (2025-01-22)

**Architecture Design Complete** ✅

- 19-module architecture approved and documented
- 7-phase roadmap from foundation to visualization
- Dependency graph validated (no circular imports)
- Critical preservation requirements finalized

**Team Status**:
- Ripley: Architecture design complete
- Dallas: Phases 1-3 implemented (16 modules, 58 KB)
- Hockley: Test infrastructure complete (4-level pyramid, 32 test suites)

**Next Steps**: Phases 4-7 implementation with parallel test development
