# Neuroevolution-ipynbs

**Frontend explicativo:** [abrir el explorador](docs/index.html) · [guía de ejecución](docs/FRONTEND.md). Incluye datos reales del proyecto, el procesamiento de audio y log-mel, construcción animada de CNN 1D/2D y evolución ilustrativa con referencias al código. Para verlo localmente, servir la raíz del repositorio y abrir `/docs/`.

Proyecto de investigación para clasificación de voz en detección de Parkinson mediante un enfoque de **neuroevolución híbrida** sobre redes **Conv1D**. 

**Estado actual**: El código se ha **refactorizado** desde notebooks monolíticos a un paquete modular `neuroevolution/` mientras se preserva exactamente la funcionalidad original. Los notebooks ahora actúan como **orquestadores ligeros** que importan y coordinan módulos reutilizables.

## Descripción general

Este repositorio implementa un pipeline híbrido que combina:

1. **Algoritmo genético** para evolucionar arquitecturas de redes Conv1D
2. **Evaluación de fitness** con 5-fold cross-validation en paralelo (ThreadPoolExecutor), usando solo validación durante la evolución y `fitness_metric="f1_score"` por defecto
3. **Entrenamiento supervisado** con técnicas adaptativas de mutación y cruce
4. **Checkpointing dinámico** del mejor modelo global según validación durante evolución
5. **Evaluación final held-out** con selección por validación y métricas test-only (Accuracy, Precision, Recall, F1, AUC, matriz de confusión)

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
- `selection.py` — Elitismo, selección por fitness y selección multiobjetivo Pareto
- `speciation.py` — Agrupación por distancia de compatibilidad, especiación

#### 5. **evolution/**
- `engine.py` — Orquestador HybridNeuroevolution (generaciones, evolución, estadísticas)
- `fitness.py` — Evaluación paralela de fitness con 5-fold CV; el fitness usa validación, no test

#### 6. **evaluation/**
- `metrics.py` — Cálculo de métricas (accuracy, precision, recall, F1, AUC, sensibilidad, especificidad)
- `cross_validation.py` — Pipeline 5-fold final: validación selecciona epoch, test reporta métricas
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

Ambos preservan la evaluación paralela de fitness con 5-fold CV y métricas agregadas. Durante la evolución, el fitness se calcula con el split de validación; el split de test se reserva para la evaluación final del mejor genoma. La selección puede funcionar en modo `fitness` para reproducir el comportamiento escalar clásico, o en modo `pareto` para priorizar tradeoffs entre F1, tiempo de evaluación y número de parámetros.

## Datos y estructura

### Organización de `data/`

```
data/
├── sets/
│   ├── folds_5/                          # 5 particiones train/validation/test (.npy files) — principal
│   ├── generated_together_train_40_1e5_N/
│   ├── test_together_N/
│   └── test_together_syn_1_N/
├── csv/                                  # Feature tables, metadata
├── control_files_short_24khz/            # Real audio (Control)
├── pathological_files_short_24khz/       # Real audio (Pathological)
└── pretrained_40_1e5_BigVSAN_generated_*/ # Synthetic data resources
```

**Principal**: Los archivos en `data/sets/folds_5/` contienen splits `.npy` para entrenamiento/validación/test.

La configuración predeterminada usa vowels: los folds internos para la búsqueda y el test global aislado para el informe final de `test.ipynb`. Los selectores de otras fuentes para cada fase se declaran explícitamente en `config.py`.

### Protocolo train/validation/test

El flujo tiene dos fases de datos:

- **Evolución genética:** `train` y `validation` sintéticos; el test sintético no se consulta para fitness.
- **Evaluación final:** el genoma ganador se reentrena desde cero con `train` real, usa `validation` real para seleccionar checkpoint y se mide una sola vez en `test` real.

En cada fase, los folds mantienen tres subconjuntos separados:

- `train`: usado para actualizar pesos del modelo.
- `validation`: usado para early stopping, selección de checkpoint y fitness evolutivo.
- `test`: reservado para la evaluación final held-out, sin monitorizar entrenamiento ni seleccionar checkpoints.

La aptitud evolutiva se calcula como el promedio del F1-score de validación a través de los folds.

Este protocolo no debe describirse como validación cruzada clásica salvo que se aporte un manifiesto que pruebe que cada sujeto aparece en exactamente un subconjunto de test a lo largo de las cinco particiones. En la redacción metodológica, usar "cinco particiones hold-out estratificadas por sujeto" o "protocolo de cinco folds train/validation/test" cuando no se disponga de ese manifiesto.

### Protocolo validación/test

- **Evolución**: train entrena cada fold y validation calcula early stopping, selección del mejor estado y fitness agregado.
- **Métrica objetivo**: `fitness_metric` controla la métrica agregada como fitness; por defecto es `f1_score`.
- **Selección de checkpoint por fold**: `fold_selection_metric="fitness_metric"` hace que el mejor estado se guarde con la misma métrica objetivo. También puede fijarse explícitamente a `accuracy`, `precision`, `sensitivity`/`recall`, `specificity`, `f1_score` o `auc`.
- **Test**: no participa en la presión evolutiva ni en la selección de epoch.
- **Evaluación final**: validation selecciona/restaura el mejor epoch de cada fold y test produce las métricas finales reportadas.
- **Migración**: runs antiguos que usaban `val + test` como fitness no son directamente comparables con este protocolo limpio.
- **Comparabilidad**: runs que seleccionan checkpoints por `accuracy` y runs que los seleccionan por `f1_score` no deben compararse como experimentos idénticos, porque pueden preservar epochs distintos.

### Selección Pareto y coste

- `selection_strategy="pareto"` rankea individuos por dominancia Pareto.
- Objetivos por defecto: maximizar `fitness`, minimizar `evaluation_time_seconds` y minimizar `parameter_count`.
- `pareto_fitness_epsilon` permite tratar diferencias muy pequeñas de fitness como equivalentes para no sobrerreaccionar a ganancias marginales caras.
- `selection_strategy="fitness"` conserva la selección histórica por fitness escalar, útil para comparaciones directas con runs antiguos.
- Runs Pareto y runs fitness-only no son directamente comparables como experimentos idénticos: cambian la presión evolutiva aunque el checkpoint global siga eligiéndose por mayor fitness escalar.

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
│       ├─> Parallel fitness eval (5 fold threads, validation-only)
│       ├─> Mutation/crossover/selection from genetics/
│       ├─> Validation-selected checkpoint save (artifacts/*)
│       └─> Progress logging
│
├── Celda 6: Evaluación final held-out en 5 particiones
│   └─> neuroevolution.evaluate_5fold_cross_validation()
│
└── Celda 7: Visualización
    └─> neuroevolution.plot_fitness_evolution()
```

**Ventaja**: Lógica centralizada, testeable, versionable. Notebook = configuración + orquestación, no implementación.

## Requisitos y configuración

### Dependencias

```bash
python -m pip install --upgrade "torch==2.11.0" "torchvision==0.26.0" "numpy>=1.21.0" "matplotlib>=3.5.0" \
            "seaborn>=0.11.0" "tqdm>=4.64.0" "jupyter>=1.0.0" "ipywidgets>=8.0.0" scikit-learn "packaging>=23.0"
```

Instala las dependencias desde una terminal antes de abrir el kernel, usando el mismo entorno de Python que Jupyter. `REQUIRED_PACKAGES` fija Torch 2.11.0 y Torchvision 0.26.0 como pareja reproducible; Torchvision 0.23.0 exige Torch 2.8.0 y es incompatible con Torch 2.11.0. Consulta la [tabla oficial de compatibilidad](https://github.com/pytorch/vision/blob/main/README.md).

Si ya tienes Torch 2.11.0 y Torchvision 0.23.0, cierra los kernels que usan ese entorno, ejecuta desde su terminal `python -m pip install --upgrade "torch==2.11.0" "torchvision==0.26.0"` y abre un kernel nuevo. El diagnóstico del notebook muestra el comando con la ruta de su intérprete para evitar reparar otro entorno.

El notebook solo comprueba el entorno y nunca ejecuta pip:
```python
from neuroevolution import verify_dependencies
verify_dependencies()
```

Los nombres antiguos `install_package`/`install_packages` se mantienen como verificadores. Si faltan dependencias, hay versiones incompatibles o un import falla, se muestra un diagnóstico y se detiene el arranque sin sustituir paquetes. Una instalación fallida sobre NFS puede dejar librerías parcialmente actualizadas: cierra todos los kernels que las usan y repara la pareja desde una terminal antes de reiniciar la ejecución.

### Una única configuración

Todos los ajustes del experimento se editan en `neuroevolution/config.py`. `test.ipynb` importa `CONFIG`; `vowels_dataset_por_persona_half.ipynb` importa `VOWELS_HALF_DATASET_CONFIG`. El notebook original conserva `VOWELS_DATASET_CONFIG` y su carpeta de salida. Los notebooks no cambian rutas, dimensiones, batch ni hiperparámetros. Los loaders rechazan discrepancias con los arrays y no buscan otra carpeta. Las tasas de mutación y los límites de profundidad actuales son estado del motor y se guardan aparte del config.

El selector `DEFAULT_DATASET = 'old_vowels'` activa `artifacts/old_vowels_dataset`, preparado por `old_vowels_dataset_por_persona.ipynb`. Incluye audios reales y generados de 0,48 s a 24 kHz: train tiene 8.609 audios, val 1.230 y test únicamente 60 reales. Se han quitado los 2.399 generados de las personas de test. Cada persona mantiene juntos sus reales y generados en cada partición y fold.

Cambia `DEFAULT_INPUT_MODALITY` en ese archivo: `audio` es el perfil predeterminado, con ondas de 11.520 muestras y directorio `arrays_audio`. El perfil `spectrogram` usa entradas de 64 × 44 y `arrays`. Ambos perfiles tienen batch 256 y referencias conocidas habilitadas: copias 1D para audio y grafos 2D para espectrogramas. Los resultados van a `artifacts/test_old_vowels_<modalidad>`. `DEFAULT_DATASET = 'vowels_half'` permite volver al dataset de 3,5 s y a su directorio de resultados propio. Reinicia el kernel después de cambiar los selectores. Consulta [la guía del dataset old](docs/old_vowels_dataset.md).

| Ajuste | Valor inicial | Propósito |
| --- | --- | --- |
| `population_size` / `max_generations` | 20 / 100 | Ampliar candidatos y horizonte de búsqueda |
| `selection_strategy` | `fitness` | Priorizar F1; Pareto sigue disponible |
| `elite_percentage` | 0.10 | Conservar al mejor y dejar espacio a descendientes |
| `base_mutation_rate` / límites | 0.20 / 0.05–0.50 | Subir mutación cuando falta diversidad o mejora |
| `crossover_rate` | 0.85 | Recombinación y conservación ocasional de padres |
| `speciation_enabled` | `True` | Seleccionar padres entre familias diferentes |
| `random_immigrant_fraction` | 0.10 | Introducir candidatos nuevos en cada generación |
| `architecture_template_seed_fraction` | 0.40 | Sembrar la población con referencias completas |
| `reference_injection_fraction` | 0.10 | Recorrer todas las referencias durante la búsqueda |
| `num_epochs` / `epoch_patience` | 100 / 25 | Entrenar cada fold con selección por validación |
| `early_stopping_generations` | 30 | Contar estancamiento una vez por generación evaluada |
| `fold_parallel_workers` | 3 | Concurrencia inicial en CUDA automático; fija en CPU o con modo automático desactivado |
| `fold_auto_workers` / `fold_auto_max_workers` | `True` / 10 | En CUDA, añadir hasta 10 folds según la VRAM |
| `fold_vram_reserve_gib` / `fold_vram_reserve_fraction` | 2 GiB / 0.10 | Reservar el mayor de 2 GiB o el 10% de la VRAM total |
| `fold_auto_growth_interval_s` | 30 | Añadir como máximo un fold cada 30 segundos |
| `fold_oom_max_retries` | 2 | Reintentar un fold con menos concurrencia si falla la memoria |
| `minimum_specificity` / `minimum_sensitivity` | 1% / 1% | Invalidar clasificadores que colapsan a una clase |
| `require_all_folds` | `True` | Evitar recompensar evaluaciones incompletas |
| `resume_evolution` | `True` | Reanudar automáticamente el progreso del experimento |

Los optimizadores, sus rangos de learning rate y sus parámetros (`weight_decay`, `momentum`, etc.) también viven en el config. Los límites de la búsqueda de CNN propias son 11 capas convolucionales, 4 capas densas y 20 millones de parámetros. Las referencias tienen su propio límite, 150 millones; nunca se recortan para ajustarlas a los límites de las CNN propias.

### Arquitecturas completas

El registro incluye **37 referencias**: LeNet-5; AlexNet; VGG-11/13/16/19 y sus variantes BN; ResNet-18/34/50/101/152; Wide ResNet-50-2/101-2; ResNeXt-50-32×4d/101-32×8d; GoogLeNet; Inception-v3; DenseNet-121/161/169/201; MobileNet-v2/v3-large; EfficientNet-B0–B7; SqueezeNet-1.0/1.1.

Se construyen desde cero con las implementaciones completas de Torchvision, conservando bloques, anchuras, capas densas y ramas auxiliares. LeNet-5 conserva C1/S2/C3/S4/C5/F6, incluidas las conexiones parciales de C3 y el subsampling entrenable. Para la tarea se adapta la entrada a un canal y la salida a dos clases; en LeNet se sustituye la salida RBF de dígitos por un clasificador entrenable. Se redimensiona el espectrograma según el config: 32 × 32 para LeNet, 299 × 299 para Inception-v3, 224 × 224 para las demás y las resoluciones 224–600 para EfficientNet-B0–B7. Las referencias completas recombinan y mutan sus hiperparámetros o cambian de arquitectura completa; los operadores no cortan arbitrariamente su grafo.

Para `audio`, `reference_architectures_1d.py` genera una copia temporal de cada
grafo: convoluciones, pooling y BatchNorm pasan a 1D, conservando canales,
grupos, profundidad, conexiones residuales/densas, ramas Inception y sus
clasificadores auxiliares. Los kernels cuadrados pasan de k×k a longitud k;
los factorizados 1×k y k×1 se convierten en convoluciones temporales independientes
de longitud k. Las cuadrículas de pooling adaptativo pasan a intervalos temporales;
se adapta la entrada de la primera capa densa, manteniendo sus neuronas ocultas.
LeNet usa pooling temporal global después de C5 para aceptar ondas de longitud
variable. Se procesa directamente `(batch, canales, muestras)`, sin convertir
la onda en imagen ni reducirla a 224 muestras. Son adaptaciones 1D inicializadas
desde cero, sin pesos preentrenados 2D. La construcción usa un grafo meta para
evitar reservar primero los pesos 2D en RAM.

Los logs muestran, por ejemplo, `reference, googlenet, 1D, complete`; el genoma
guarda `reference_dimensions` para distinguir modelos y checkpoints 1D/2D.
Con población 20 y fracción inicial 0.40 se siembran siete referencias, incluida
GoogLeNet, más las CNN aleatorias. Al reanudar una población anterior se conserva
su progreso y se introducen referencias en generaciones posteriores mediante
`reference_injection_fraction` (dos por generación con los valores actuales).

Las antiguas plantillas reducidas `*_small`/`*_medium` siguen disponibles para compatibilidad, pero no son las semillas predeterminadas. El informe de una referencia muestra su grafo real. Consulta las [implementaciones de Torchvision](https://docs.pytorch.org/vision/stable/models.html), las [resoluciones de EfficientNet](https://docs.pytorch.org/vision/2.0/_modules/torchvision/models/efficientnet.html) y el [artículo de LeNet-5](https://bottou.org/papers/lecun-98h).

### Ejecutar

1. Edita únicamente `neuroevolution/config.py`.
2. Genera los datos con `vowels_dataset_por_persona_half.ipynb` si aún no existen.
3. Reinicia el kernel y ejecuta `test.ipynb` en orden.

El perfil vowels usa folds internos de las personas de train. El conjunto global de validación monitoriza el checkpoint elegido; no entra en el fitness. El test global se carga tras elegir la arquitectura. El checkpoint es el mejor fold del genoma elegido, por lo que su F1 de monitorización puede diferir del F1 medio de los cinco folds.

Por defecto, la ejecución busca `evolution_progress.json` en `CONFIG['artifacts_dir']` (por ejemplo, `artifacts/test_old_vowels_audio`) y restaura la población, el mejor individuo, el historial, el contador de estancamiento y los parámetros de evolución. Si la generación guardada ya terminó, continúa con la siguiente; si la población está pendiente de evaluación, retoma esa generación. `max_generations` es el límite total, contando las generaciones anteriores: aumentarlo permite continuar un run que alcanzó ese límite, siempre que no haya alcanzado el objetivo de fitness o el límite de estancamiento.

Para trasladar un run, copia el JSON y su checkpoint `.pth` a la carpeta del experimento: las rutas del checkpoint se resuelven en la carpeta actual y la carpeta del dataset puede cambiar de ubicación conservando su nombre e identidad. El JSON por sí solo permite recuperar la evolución, pero no contiene los pesos entrenados ni el estado de un entrenamiento interrumpido dentro de un fold. La configuración de entrenamiento y el tamaño de población deben ser compatibles; si el progreso está dañado o es incompatible, la ejecución se detiene conservando los archivos anteriores. Los JSON antiguos sin snapshot requieren comenzar una ejecución nueva. Para empezar de cero, usa otra carpeta de artifacts o establece explícitamente `resume_evolution=False`.

```python
from neuroevolution import CONFIG, setup_device_and_seeds, HybridNeuroevolution
from neuroevolution.config import configure_vowels_input, validate_config

validate_config(CONFIG)
configure_vowels_input(CONFIG)
device = setup_device_and_seeds(CONFIG['seed'])
evolver = HybridNeuroevolution(CONFIG, device)
best_genome = evolver.evolve()
```

## Artefactos generados

Después de ejecutar, se crean en `artifacts/{best_audio|test_audio}/`:

```
artifacts/best_audio/
├── execution_log.txt              # Todas las salidas de print()
├── generation_progress.txt        # Resumen legible por generación
├── evolution_progress.json        # Estadísticas JSON (fitness, Pareto, coste, diversity, etc.)
├── fold_progress.json             # Estado, época y fitness de cada fold durante la generación
├── best_model_checkpoint.pth      # Checkpoint del mejor global seleccionado por validación
├── final_cv_results.json          # Métricas finales held-out de test (accuracy, F1, AUC, matriz)
├── fitness_evolution.png          # Gráfica de evolución de fitness
├── architecture_summary.txt       # Arquitectura del mejor modelo
└── [otros reportes y visualizaciones]
```

Durante la evaluación, `fold_progress.json` se actualiza al comenzar cada fold,
tras la primera época y después cada cinco épocas, y al terminar. `running`
indica que el fold está asignado; `started_at` confirma que el trabajador
empezó; `epoch` muestra entrenamiento completado. `fitness` permanece en
`null` hasta que termina el fold y `average_fitness` hasta que terminan los
cinco folds del individuo. El log emite una señal cada 30 segundos mientras
espera resultados. En otra terminal del servidor se pueden consultar:

```bash
tail -f artifacts/test_vowels_audio/execution_log.txt
nvidia-smi
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
- ✅ Selección Pareto: dominancia, frentes, crowding distance y torneo cost-aware
- ✅ Entrenamiento del modelo (forward pass, backward, update)
- ✅ Evaluación paralela (5-fold ThreadPoolExecutor)
- ✅ Separación validation/test: fitness solo con validación y test solo en evaluación final
- ✅ Selección de checkpoint por métrica configurable: F1 por defecto o accuracy explícita para compatibilidad
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

El protocolo por defecto es:

- **Búsqueda sintética → evaluación real**: evolución con `files_syn_all_N`; después, reentrenamiento y métricas finales con `files_real_N`. Es el flujo implementado en `evaluate_5fold_cross_validation()`.

También permite experimentar con:

- **Real-only**: entrenamiento, validación y test con audios reales (`files_real_N`).
- **Synthetic-only**: entrenamiento, validación y test con datos sintéticos; no produce métricas finales reales.
- **Mixed**: mezcla real+sintética. Exploratorio si validation/test contienen sintéticos.
- **Generalization**: entrenar con sintéticos y testear en real. Requiere regenerar o documentar folds donde validation y test sean reales y los sintéticos procedan solo de sujetos de entrenamiento.

Configurable mediante los selectores `evolution_*` y `final_*` en CONFIG.

## Notas para investigadores

1. **Reproducibilidad**: Todos los seeds se fijan al inicio. `SEED=42` por defecto.

2. **Checkpointing**: El mejor modelo se guarda dinámicamente según F1-score de validación. Interrumpir (Ctrl+C) es seguro—el mejor hasta ese momento está en `.pth`.

3. **Memoria GPU**: La cola de evolución en CUDA ajusta la concurrencia automáticamente.
   Consulta la VRAM libre del dispositivo CUDA seleccionado (también su partición MIG),
   empieza con hasta `fold_parallel_workers` folds si caben y aumenta de uno en uno
   cuando queda margen para otro fold además de la reserva. Presupuesta también
   los folds que están arrancando y aún no han completado su primera época.
   Usa una estimación inicial de 4 GiB por fold y la memoria observada durante
   los últimos 60 segundos, con un factor de seguridad de 1.5.
   La reserva es preventiva: las arquitecturas cambian y pueden tener picos mayores.
   Si ocurre un OOM o el fallo de NVML del allocator, deja terminar los folds activos,
   libera caché y reintenta el fold con un límite menor. Puede volver a crecer
   tras una espera de al menos 30 segundos (o dos intervalos de crecimiento)
   si la memoria lo permite. Si agota los reintentos, registra el error y asigna
   0% al fold, y la evolución continúa. Con concurrencia fija se asigna 0% directamente.
   Con `require_all_folds=True` (por defecto), un fold fallido deja en cero el fitness
   del individuo. Los errores ajenos a memoria mantienen
   su tratamiento anterior. La telemetría se guarda en `fold_progress.json` y los
   cambios de concurrencia aparecen en el log. Estas opciones se aplican a la cola
   global del motor evolutivo, no a la evaluación final ni al replay del ganador.
   Para concurrencia fija, usa `fold_auto_workers=False` y `fold_parallel_workers=N`.
   Las copias de los mejores pesos y los modelos de folds terminados se guardan
   en CPU; los futuros pendientes de procesar no conservan modelos en GPU.
   Los datasets cacheados también están en CPU. PyTorch puede conservar VRAM
   reservada para reutilizarla: `reserved_vram_bytes` incluye esa caché y
   `allocated_vram_bytes` mide los tensores vivos. No se vacía la caché después
   de cada fold; tras un error de memoria se hace cuando terminan los folds activos.
   Mientras espera a esos folds, el estado es `draining_after_memory_error`:
   siguen actualizándose la VRAM y el tiempo de cooldown. Los mensajes periódicos
   muestran el ID, fold y época de los entrenamientos activos. La cola espera a
   que terminen antes de liberar caché y reintentar; superar el cooldown no
   interrumpe un fold que sigue entrenando. `generation_progress.txt` registra
   cada individuo cuando finalizan sus cinco folds; `fold_progress.json` permite
   seguir las épocas antes de que haya resultados completos por individuo.
   Si un fold aislado no cabe, reduce `batch_size` o utiliza una partición MIG mayor.
   El perfil de audio usa `batch_size=256`. Usar 2
   multiplica aproximadamente por 128 los lotes de cada época y puede dejar gran
   parte de la VRAM libre sin mejorar el rendimiento. La cola registra el tamaño
   de lote efectivo, las épocas, la paciencia y AMP al comenzar cada generación.

4. **Logs detallados**: Todo `print()` se guarda en `execution_log.txt`. Revisar ahí si falla evolution.

5. **Variantes algorítmicas**: Cambiar entre `best` (clásico) y `test` (NEAT-like) es solo cambiar notebook. El paquete `neuroevolution/` soporta ambas configuraciones.

## Estatus del proyecto

- ✅ Refactorización completada (19+ módulos, 7 paquetes)
- ✅ Notebooks actualizados a orquestadores ligeros
- ✅ Tests operacionales y regresiones de protocolo validation/test
- ✅ Documentación refrescada
- 🔲 Validación en CI/CD (próximo fase)

## Licencia

Este proyecto esta publicado bajo licencia **MIT**. Consulta `LICENSE` para mas detalles.
