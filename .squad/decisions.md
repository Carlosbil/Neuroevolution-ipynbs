# Squad Decisions

## Active Decisions

### Bishop: Fold Rotation Rebuild (2026-04-21)
**Status:** Completed

Evolution evaluation strategy changed from all-fold parallel training to single rotating fold per generation:
- Fold selection deterministic: `(generation % num_folds) + 1`
- Population evaluation in fixed 5-individual chunks with parallel training per chunk
- Mutation growth bias made explicit and generation-aware (`current_growth_bias`)
- Removed double-cap injected individual and reserve-slot logic
- Backward-compatible via config flags; `evaluate_fitness` remains compatible

**Impact:** Reduces per-individual compute load, improves generation throughput, maintains architectural flexibility.

### Hicks: New Rotating-Fold Notebook Setup (2026-04-21)
**Status:** Completed

New notebook implementation preserving test.ipynb while adding explicit rotating-fold scheduling:
- Location: `mejoras/reimplementacion_rotating_fold/rotating_fold_chunked_evolution.ipynb`
- Artifacts: `artifacts/rotating_fold_chunked_audio/`
- Fold rotation and chunking explicit and reproducible
- Canonical run flow maintained (imports → config → verify → evolve → report)

**Impact:** Enables clean reimplementation workflow, improves experiment traceability without overwriting existing notebooks.

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction
