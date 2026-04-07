# Neuroevolution Audio Classification Project

Research project for Parkinson voice detection using **hybrid neuroevolution** on **Conv1D** networks. Primary workflow is in Jupyter notebooks for experimentation, traceability, and rapid configuration comparison.

## Project Context

This is an academic research project focused on finding robust architectures to classify audio signals as **Control vs Pathological** (Parkinson detection). The system evolves neural network architectures using genetic algorithms combined with supervised training.

## Dependencies

Install the following packages to run the notebooks:

```bash
pip install torch==2.11.0 torchvision==0.26.0 numpy>=1.21.0 matplotlib>=3.5.0 seaborn>=0.11.0 tqdm>=4.64.0 jupyter>=1.0.0 ipywidgets>=8.0.0 scikit-learn
```

The notebooks will auto-install missing packages on first run.

## Notebook Structure

### Three Main Notebooks

1. **`best_Audio_hybrid_neuroevolution_notebook.ipynb`** - Main reference notebook
   - Uses `artifacts/best_audio/` for outputs
   - Classic genome representation (layers, filters, kernels, optimizers)
   - Standard adaptive mutation and crossover
   - For final production runs and results

2. **`test.ipynb`** - Experimental iteration notebook
   - Uses `artifacts/test_audio/` for outputs
   - Advanced algorithm variant with:
     - Innovation-based genome tracking (`innovation_uuid`, `innovation_genes`, `structural_history`)
     - NEAT-like crossover with gene alignment
     - Genetic speciation with compatibility distance
     - Incremental complexity growth with dynamic limits
   - For testing new evolutionary strategies

3. **`test_simplified.ipynb`** - Reorganized version of `test.ipynb`
   - Same logic, better readability
   - Easier to follow algorithm flow

### Notebook Execution

Run cells sequentially:
1. Imports and environment setup
2. Configuration (`CONFIG` dict)
3. Data loading/verification
4. Neuroevolution (`neuroevolution.evolve()`)
5. Final 5-fold evaluation with complete metrics

Before running, verify:
- `CONFIG['data_path']` points to `data/sets/folds_5`
- `CONFIG['dataset_id']` and `CONFIG['fold_id']` are set correctly
- `info_path` variable matches intended artifact directory

## Key Architectural Concepts

### Hybrid Neuroevolution Pipeline

The system combines genetic algorithms with supervised training:

1. **Population initialization**: Random Conv1D architectures with varying depth, filters, kernel sizes
2. **Parallel fitness evaluation**: Each individual evaluated on all 5 folds simultaneously using `ThreadPoolExecutor`
3. **Selection**: Elitism (20%) + fitness-proportional selection
4. **Genetic operators**: Crossover (99%) and adaptive mutation (10-80%)
5. **Convergence**: Stop on fitness threshold or max generations

### Genome Representation

Classic (`best` notebook):
- Conv layers: count, filters per layer, kernel sizes
- FC layers: count, nodes per layer
- Hyperparameters: optimizer, learning rate, dropout
- Normalization: batch/layer norm choices

Advanced (`test` notebooks):
- All of the above plus:
- `innovation_uuid`: unique architecture identifier
- `innovation_genes`: gene-level innovation IDs for alignment
- `structural_history`: mutation trace for debugging

### Conv1D Architecture Generation

Generated architectures follow this structure:
```
Input (1, sequence_length) 
→ [Conv1D + Norm + Activation + Dropout] × N
→ Adaptive pooling → Flatten
→ [FC + Activation + Dropout] × M
→ Output (2 classes)
```

Key constraints:
- 1-30 conv layers, 1-10 FC layers
- Filters: 1-256, FC nodes: 64-1024
- Kernel sizes: [1, 3, 5, 7, 9, 11, 13, 15]
- Dropout: 0.2-0.6

### Parallel 5-Fold Cross-Validation

**During evolution**: Each individual trains on all 5 folds in parallel (separate threads). Fitness = average accuracy across folds. This is fast and robust against overfitting.

**Final evaluation**: Best architecture re-evaluated with full metrics (accuracy, sensitivity, specificity, precision, F1, AUC, confusion matrix).

## Data Organization

### Directory Structure

```
data/
├── sets/
│   ├── folds_5/                           # 5-fold CV splits (.npy files)
│   ├── generated_together_train_40_1e5_N/ # Synthetic training data
│   ├── test_together_N/                   # Real test set
│   └── test_together_syn_1_N/             # Synthetic test set
├── csv/                                   # Feature tables and metadata
├── control_files_short_24khz/             # Real control audio
├── pathological_files_short_24khz/        # Real pathological audio
├── pretrained_40_1e5_BigVSAN_generated_control/
└── pretrained_40_1e5_BigVSAN_generated_pathological/
```

### Data Scenarios

The project supports multiple training/test combinations:
- Real only
- Synthetic only  
- Real + Synthetic mixed
- Train synthetic, test real (generalization test)

Configure via `dataset_id` and `fold_id` in `CONFIG`.

### Data Format

All datasets are `.npy` files with structure:
- `X_train`, `y_train`: training samples and labels
- `X_val`, `y_val`: validation samples and labels
- `X_test`, `y_test`: test samples and labels

Audio waveforms are 1D arrays (typically ~240,000 samples at 24kHz).

## Artifact Outputs

Generated during and after evolution:

```
artifacts/{best_audio|test_audio}/
├── evolution_progress.json        # Generation-by-generation stats
├── generation_progress.txt        # Human-readable progress log
├── execution_log.txt              # All print() output during run
├── best_model_checkpoint.pth      # Best global model (updated dynamically)
└── [plots and metric tables]      # Generated at end of run
```

## Configuration Patterns

### CONFIG Dictionary Structure

All notebooks use a `CONFIG` dict with these sections:

1. **Genetic algorithm**: `population_size`, `max_generations`, `fitness_threshold`, `elite_percentage`
2. **Mutation/crossover**: `base_mutation_rate`, `mutation_rate_min/max`, `crossover_rate`
3. **Architecture bounds**: `min/max_conv_layers`, `min/max_fc_layers`, `min/max_filters`, `min/max_fc_nodes`
4. **Training**: `num_epochs`, `learning_rate`, `batch_size`, early stopping thresholds
5. **Data**: `data_path`, `dataset_id`, `fold_id`, `num_channels`, `num_classes`
6. **Artifacts**: `artifact_dir` (where to save outputs)

### Typical Adjustments

When experimenting, commonly modified params:
- `population_size`: 10-50 (tradeoff: diversity vs speed)
- `max_generations`: 50-200
- `elite_percentage`: 0.1-0.3 (how many top individuals to preserve)
- `base_mutation_rate`: 0.2-0.4
- `max_conv_layers`, `max_fc_layers`: control search space complexity
- `dataset_id`, `fold_id`: switch between data scenarios

## Advanced Features (test.ipynb only)

### Innovation Tracking

Each genome gets a `innovation_uuid` to trace lineage. Genes have `innovation_id` for alignment during crossover (NEAT-style).

### Genetic Speciation

Individuals grouped into species based on compatibility distance (structural + parametric differences). Each species:
- Has its own survival threshold
- Contributes to next generation proportionally
- Prevents premature convergence

### Incremental Complexity Growth

Architecture complexity increases gradually:
- `current_max_conv_layers`, `current_max_fc_layers` start low, grow per generation
- `incremental_growth_probability`: bias toward adding layers
- Prevents bloat early in evolution

## Code Conventions

### Logging System

All `print()` calls are redirected to `artifacts/{dir}/execution_log.txt` via a custom logger. This ensures reproducibility and full execution traces.

### Seed Management

Fixed seeds for reproducibility:
```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
```

### Threading for Fold Evaluation

`ThreadPoolExecutor` with `max_workers=5` evaluates folds in parallel:
```python
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(train_fold, genome, fold_i) for fold_i in range(5)]
    results = [future.result() for future in as_completed(futures)]
```

### Device Configuration

Auto-detects CUDA:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

All models and tensors moved to `device` during training.

## Common Issues

### Audio Sequence Length Mismatch

If loading different datasets, `sequence_length` may vary. The notebooks auto-detect on first load, but verify `CONFIG['sequence_length']` matches loaded data shape.

### Memory Issues with Parallel Fold Training

5 simultaneous models can exhaust GPU memory. If OOM errors occur:
- Reduce `batch_size`
- Reduce `max_workers` in `ThreadPoolExecutor` (e.g., 3 instead of 5)
- Use CPU training

### KeyboardInterrupt During Evolution

Evolution can be interrupted with Ctrl+C. The best model so far is always saved in `best_model_checkpoint.pth`.

### Artifacts Directory Conflicts

`best` and `test` notebooks use different `info_path` dirs to avoid collisions. If reusing artifact paths, previous files may be overwritten.

## Language Note

README and notebook documentation are primarily in Spanish (academic project context). Code comments and variable names are mixed Spanish/English.
