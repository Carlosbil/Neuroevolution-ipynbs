# Skill: Notebook Orchestration Smoke Validation

**Pattern**: Validate refactored notebook orchestration quickly by executing import contracts and runner scripts in subprocesses before expensive training tests.

## When to use

- Notebook was refactored to call `neuroevolution` modules.
- You need strict “no behavior change” confidence fast.
- Full evolution runs are too expensive for every QA cycle.

## Steps

1. **Top-level import smoke**
   - Run `python -c "import neuroevolution"` from repo root.
   - Fails early on broken package wiring/export paths.

2. **Notebook import-cell smoke**
   - Parse `test_simplified.ipynb`.
   - Execute the orchestration import code cell in a subprocess.
   - Confirms notebook/module integration contracts without full training.

3. **Script smoke**
   - Run `python test_phases_1_3.py`.
   - If it fails on Windows encoding, retry with `PYTHONUTF8=1` to isolate logic-vs-console issues.

4. **Promote to automated tests**
   - Add integration tests under `tests/integration/` for all three checks.
   - Keep failures file-referenced for fast handoff to implementation.

5. **Gate on regression baselines**
   - Assert required files exist in `validation_artifacts/reference/` before claiming equivalence.
   - Example required artifacts: `fold_0_data.npz`, `100_random_genomes.json`, `1000_mutations.json`, `100_crossover_children.json`, `mini_evolution_progress.json`, `final_cv_results.json`.

## Key paths

- `test_simplified.ipynb`
- `test_phases_1_3.py`
- `tests/integration/test_orchestration_smoke.py`
