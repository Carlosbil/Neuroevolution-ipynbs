# Project Context

- **Owner:** Carlosbil
- **Project:** Reimplement test notebook and scripts from scratch
- **Stack:** Python, PyTorch, Jupyter notebooks
- **Created:** 2026-04-21

## Learnings

- Main implementation target is a new notebook and new folder, independent from current test.ipynb.
- Evolution flow must rotate folds generation-by-generation instead of full 5-fold per individual.
- Mutation policy should bias toward larger architectures each generation.
- Evolution engine now evaluates one rotating fold per generation, with deterministic fold selection and 5-individual chunked parallel population evaluation.

## Work Completed (2026-04-21)

**Task:** Rebuild evolution strategy for rotating single-fold + 5-individual chunks + growth-biased mutation + remove double-cap injected individual

**Files Modified:**
- `neuroevolution/config.py` — Added config params for fold rotation and chunk size
- `neuroevolution/evolution/engine.py` — Implemented rotating fold scheduling and chunked evaluation
- `neuroevolution/evolution/fitness.py` — Added `active_fold` parameter for single-fold fitness evaluation
- `neuroevolution/genetics/mutation.py` — Implemented generation-aware growth bias for larger architectures

**Key Changes:**
1. Fold rotation deterministic via `(generation % num_folds) + 1` mapping
2. Population evaluation in fixed 5-individual chunks evaluated in parallel
3. Growth bias now explicit and generation-aware, biasing mutation toward larger architectures
4. Removed double-cap injected individual and reserve-slot logic from initialization and reproduction
5. Backward-compatible via config flags; existing API unchanged
