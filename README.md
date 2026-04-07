# Neuroevolution-ipynbs

Proyecto de investigación para clasificación de voz en detección de Parkinson mediante un enfoque de **neuroevolución híbrida** sobre redes **Conv1D**. 

**Estado actual**: El código se ha **refactorizado** desde notebooks monolíticos a un paquete modular `neuroevolution/` mientras se preserva exactamente la funcionalidad original. Los notebooks ahora actúan como **orquestadores ligeros** que importan y coordinan módulos reutilizables.

## Descripción general

Este repositorio implementa un pipeline híbrido que combina:

1. **Algoritmo genético** para evolucionar arquitecturas de redes Conv1D
2. **Evaluación de fitness** con 5-fold cross-validation en paralelo (ThreadPoolExecutor)
3. **Entrenamiento supervisado** con técnicas adaptativas de mutación y cruce
4. **Checkpointing dinámico** del mejor modelo global durante evolución
5. **Evaluación final** con métricas completas (Accuracy, Precision, Recall, F1, AUC, matriz de confusión)

El objetivo es encontrar arquitecturas robustas para separar clases **Control vs Pathological** en señales de voz.

## Estructura del proyecto (post-refactorización)

### Notebooks: Orquestadores ligeros

- **`best_Audio_hybrid_neuroevolution_notebook.ipynb`** — Referencia estable
  - Representa el flujo clásico de neuroevolución
  - Usa `artifacts/best_audio/` para guardar resultados
  - Punto de partida para runs de producción
  
- **`test.ipynb`** — Histórico (mantener como referencia)
  - Variante con algoritmo NEAT-like avanzado (innovación, especiación)
  - Usa `artifacts/test_audio/` para logs separados

### Paquete `neuroevolution/`: Módulos reutilizables

Organizado en **7 submódulos** (26 archivos .py, ~78 KB):

#### 1. **Core** (raíz)
- `config.py` — Diccionario CONFIG con 32+ parámetros de experimento
- `device_utils.py` — Detección CUDA, configuración de device único
- `logger.py` — Redirección de print() a archivo de log
- `__init__.py` — Importaciones estables y backward-compatible aliases

#### 2. **data/**
- `loader.py` — Carga de folds (.npy), fallback en múltiples rutas, validación de secuencia

#### 3. **models/**
- `evolvable_cnn.py` — Arquitectura Conv1D generada dinámicamente a partir del genoma
- `genome_validator.py` — Validación de genomas (3-tier: prevención → corrección → runtime)

#### 4. **genetics/**
- `innovation.py` — Tracking NEAT-like con UUID (innovación_uuid, innovation_genes, structural_history)
- `genome.py` — Crear genomas aleatorios válidos
- `mutation.py` — Mutación adaptativa (tasa ajusta según diversidad poblacional)
- `crossover.py` — Cruce alineado por innovación (estilo NEAT)
- `selection.py` — Elitismo + selección proporcional a fitness
- `speciation.py` — Agrupación por distancia de compatibilidad, especiación

#### 5. **evolution/**
- `engine.py` — Orquestador HybridNeuroevolution (generaciones, evolución, estadísticas)
- `fitness.py` — Evaluación paralela de fitness con 5-fold CV (ThreadPoolExecutor max_workers=5)

#### 6. **evaluation/**
- `metrics.py` — Cálculo de métricas (accuracy, precision, recall, F1, AUC, sensibilidad, especificidad)
- `cross_validation.py` — Pipeline 5-fold con gestión de folds individuales
- `artifacts.py` — Gestor de artefactos (generación de checkpoints, progreso JSON, logs)

#### 7. **visualization/**
- `plots.py` — Gráficas de evolución, estadísticas, análisis de fallos
- `reports.py` — Resumen de arquitectura, info de checkpoint

## Diferencias algorítmicas: `best` vs `test`

Ambos notebooks operan sobre el mismo paquete `neuroevolution/`, pero usan configuraciones distintas:

### `best_Audio_hybrid_neuroevolution_notebook.ipynb` (Clásico)
- Representación simple del genoma (capas, filtros, kernels, FC, optimizer)
- Mutación adaptativa estándar sobre topología e hiperparámetros
- Crossover convencional
- Selección global con elitismo clásico
- **Estable y directo para runs de referencia**

### `test.ipynb` (Avanzado NEAT-like)
- Trazabilidad con `innovation_uuid`, `innovation_genes`, `structural_history`
- Mutación con crecimiento incremental de complejidad (`current_max_conv_layers`, `current_max_fc_layers`)
- Crossover alineado por innovación (empareja genes homólogos)
- Especiación genética (agrupación por distancia de compatibilidad)
- **Sofisticado: menor riesgo de convergencia prematura, mayor diversidad**

Ambos preservan la evaluación paralela de fitness con 5-fold CV y metrics agregadas.

## Datos y estructura

### Organización de `data/`

```
data/
├── sets/
│   ├── folds_5/                          # 5-fold CV (.npy files) — principal
│   ├── generated_together_train_40_1e5_N/
│   ├── test_together_N/
│   └── test_together_syn_1_N/
├── csv/                                  # Feature tables, metadata
├── control_files_short_24khz/            # Real audio (Control)
├── pathological_files_short_24khz/       # Real audio (Pathological)
└── pretrained_40_1e5_BigVSAN_generated_*/ # Synthetic data resources
```

**Principal**: Los archivos en `data/sets/folds_5/` contienen los splits 5-fold CV (.npy) para entrenamiento/validación/test.

### Estructura global (post-refactorización)

```
Neuroevolution-ipynbs/
├── neuroevolution/                       # Paquete modular (26 archivos .py, 7 módulos)
│   ├── config.py                         # CONFIG dict (32+ parámetros)
│   ├── device_utils.py                   # CUDA setup, device único
│   ├── logger.py                         # Log redirection
│   ├── data/loader.py                    # Load folds, path fallback
│   ├── models/{evolvable_cnn.py, genome_validator.py}
│   ├── genetics/{innovation.py, genome.py, mutation.py, crossover.py, selection.py, speciation.py}
│   ├── evolution/{engine.py, fitness.py}
│   ├── evaluation/{metrics.py, cross_validation.py, artifacts.py}
│   └── visualization/{plots.py, reports.py}
├── best_Audio_hybrid_neuroevolution_notebook.ipynb   # Orquestador (ref)
├── test.ipynb                            # Orquestador (NEAT-like, histórico)
├── pytest.ini                            # Test configuration
├── README.md
├── LICENSE
└── data/                                 # Datasets
```

## Flujo de ejecución: orquestación con módulos

Aunque los notebooks parecen ejecutar monolíticamente, **internamente coordinan módulos independientes**:

### Patrón: Orquestador ligero

```
Notebook (7 celdas):
├── Celdas 1-2: Imports, config, logging setup
│   └─> neuroevolution.setup_device_and_seeds()
│   └─> neuroevolution.setup_notebook_logging()
│
├── Celda 3: Carga datos
│   └─> neuroevolution.load_dataset()
│
├── Celdas 4-5: Evolución
│   └─> neuroevolution.HybridNeuroevolution(...).evolve()
│       ├─> Parallel fitness eval (5 fold threads)
│       ├─> Mutation/crossover/selection from genetics/
│       ├─> Checkpoint save (artifacts/*)
│       └─> Progress logging
│
├── Celda 6: Evaluación final 5-fold
│   └─> neuroevolution.evaluate_5fold_cross_validation()
│
└── Celda 7: Visualización
    └─> neuroevolution.plot_fitness_evolution()
```

**Ventaja**: Lógica centralizada, testeable, versionable. Notebook = configuración + orquestación, no implementación.

## Requisitos y configuración

### Dependencias

```bash
pip install torch==2.11.0 torchvision==0.26.0 numpy>=1.21.0 matplotlib>=3.5.0 \
            seaborn>=0.11.0 tqdm>=4.64.0 jupyter>=1.0.0 ipywidgets>=8.0.0 scikit-learn
```

O simplemente en el notebook:
```python
from neuroevolution import verify_dependencies, install_package
verify_dependencies()  # Auto-instala si faltan paquetes
```

### CONFIG: parámetros clave

En ambos notebooks se define un diccionario `CONFIG` con ~32 parámetros:

**Algoritmo genético**:
- `population_size` (default 20): Individuos por generación
- `max_generations` (default 100): Límite de generaciones
- `fitness_threshold` (default 98.0): Detener si se alcanza
- `elite_percentage` (default 0.2): Preservar mejores individuos

**Mutación/Crossover**:
- `base_mutation_rate` (default 0.3)
- `mutation_rate_min/max` (0.1-0.8): Rango adaptativo
- `crossover_rate` (default 0.99)

**Arquitectura (rangos)**:
- `min/max_conv_layers` (1-30)
- `min/max_fc_layers` (1-10)
- `min/max_filters` (1-256)
- `min/max_fc_nodes` (64-1024)
- `kernel_sizes` ([1, 3, 5, 7, 9, 11, 13, 15])

**Entrenamiento**:
- `num_epochs` (default 50)
- `learning_rate` (default 0.001)
- `batch_size` (default 32)
- Early stopping thresholds

**Datos**:
- `data_path`: Ruta base a `data/sets/folds_5`
- `dataset_id`, `fold_id`: Selector de escenario
- `num_channels` (1), `num_classes` (2), `sequence_length` (auto-detectado)

**Artefactos**:
- `artifact_dir`: Ruta de salida (ej. `artifacts/best_audio`)

Ver `neuroevolution.get_default_config()` para lista completa.

## Cómo ejecutar

### Opción 1: Notebook clásico (best)

```python
# En best_Audio_hybrid_neuroevolution_notebook.ipynb
CONFIG = neuroevolution.get_default_config(info_path="artifacts/best_audio")
CONFIG['data_path'] = "data/sets/folds_5"
CONFIG['dataset_id'] = "40_1e5_N"  # O tu escenario
CONFIG['fold_id'] = 0

# Ejecutar celdas en orden:
# 1. Imports + setup
# 2. Config + load data
# 3. Evolve
# 4. Final eval 5-fold
# 5. Visualize
```

### Opción 2: Desde Python scripts

```python
import torch
from neuroevolution import (
    setup_device_and_seeds, 
    get_default_config,
    load_dataset,
    HybridNeuroevolution,
    evaluate_5fold_cross_validation
)

# Setup
device = setup_device_and_seeds(seed=42)
config = get_default_config(info_path="artifacts/my_run")
config['population_size'] = 30

# Load data
dataset = load_dataset(config)

# Evolve
evolver = HybridNeuroevolution(config=config, device=device)
best_genome, progress = evolver.evolve(dataset)

# Final eval
results = evaluate_5fold_cross_validation(
    best_genome=best_genome,
    dataset=dataset,
    config=config,
    device=device
)
```

## Artefactos generados

Después de ejecutar, se crean en `artifacts/{best_audio|test_audio}/`:

```
artifacts/best_audio/
├── execution_log.txt              # Todas las salidas de print()
├── generation_progress.txt        # Resumen legible por generación
├── evolution_progress.json        # Estadísticas JSON (fitness, diversity, etc.)
├── best_model_checkpoint.pth      # Checkpoint del mejor global (dinámico)
├── final_cv_results.json          # Métricas finales (accuracy, F1, AUC, matriz)
├── fitness_evolution.png          # Gráfica de evolución de fitness
├── architecture_summary.txt       # Arquitectura del mejor modelo
└── [otros reportes y visualizaciones]
```

## Testing (post-refactorización)

El proyecto incluye infraestructura de tests:

```bash
# Ver tests disponibles
pytest --collect-only

# Ejecutar todos los tests
pytest -v

# Solo tests unitarios
pytest -v -m unit

# Solo tests de integración
pytest -v -m integration

# Tests de regresión (comparan vs artefactos de referencia)
pytest -v -m regression
```

Tests validan:
- ✅ Carga de datos (diferentes folds, fallback de rutas)
- ✅ Validación de genomas (3-tier prevention/fix/runtime)
- ✅ Operadores genéticos (mutation, crossover, selection)
- ✅ Entrenamiento del modelo (forward pass, backward, update)
- ✅ Evaluación paralela (5-fold ThreadPoolExecutor)
- ✅ Checkpointing (save/load state_dict, genome, config)
- ✅ Equivalencia numérica (tolerancias: ±1e-7 float32, ±1e-5 CUDA)

## Decisiones arquitectónicas

### 1. Patrón Orquestador Ligero

Los notebooks NO contienen lógica de algoritmo, solo coordinación. Facilita:
- Testabilidad (módulos importables, sin cell dependencies)
- Reutilización (scripts pueden importar desde neuroevolution.*)
- Mantenibilidad (cambios centralizados en módulos)

### 2. Namespace de innovación NEAT-like

Cada genoma tiene `innovation_uuid` inmutable. Genes alineables por `innovation_id`. Permite:
- Crossover inteligente (empareja genes homólogos)
- Trazabilidad estructural (historia de mutaciones)
- Especiación (distancia de compatibilidad)

### 3. Evaluación paralela con ThreadPoolExecutor

5 workers simultáneos (1 por fold), no Process pool. Razones:
- Memoria compartida (no hay serialización de tensores)
- Determinismo relativo (mismo seed = mismo orden CPU)
- Mejor para GPU (evita overhead de pickling)

### 4. Validación 3-tier

```
Génesis → Prevención (create_random_genome)
Evolución → Corrección (validate_and_fix_genome)
Runtime → Detección (is_genome_valid)
```

Evita fallos silenciosos y trazabilidad total de corrupciones.

### 5. Device único desde el inicio

`setup_device_and_seeds()` establece:
- Device (CUDA vs CPU)
- Seeds (torch, numpy, random) con SEED=42
- Logging redirection

Una sola llamada = estado reproducible.

## Escenarios de datos

El proyecto permite experimentar con:

- **Real-only**: Entrenamiento y test con audios reales
- **Synthetic-only**: Entrenamiento con datos sintéticos
- **Mixed**: Entrenamiento en mezcla real+sintética
- **Generalization**: Entrenar sintético, testear real

Configurable vía `dataset_id` y `fold_id` en CONFIG.

## Notas para investigadores

1. **Reproducibilidad**: Todos los seeds se fijan al inicio. `SEED=42` por defecto.

2. **Checkpointing**: El mejor modelo se guarda dinámicamente. Interrumpir (Ctrl+C) es seguro—el mejor hasta ese momento está en `.pth`.

3. **Memoria GPU**: Si OOM al evaluar 5 folds en paralelo:
   - Reducir `batch_size`
   - Reducir `max_workers` en `ThreadPoolExecutor` (en fitness.py)
   - Usar CPU (`device = "cpu"`)

4. **Logs detallados**: Todo `print()` se guarda en `execution_log.txt`. Revisar ahí si falla evolution.

5. **Variantes algorítmicas**: Cambiar entre `best` (clásico) y `test` (NEAT-like) es solo cambiar notebook. El paquete `neuroevolution/` soporta ambas configuraciones.

## Estatus del proyecto

- ✅ Refactorización completada (19+ módulos, 7 paquetes)
- ✅ Notebooks actualizados a orquestadores ligeros
- ✅ Tests operacionales (32 suites, 4 niveles de validación)
- ✅ Documentación refrescada
- 🔲 Validación en CI/CD (próximo fase)

## Licencia

Este proyecto esta publicado bajo licencia **MIT**. Consulta `LICENSE` para mas detalles.
