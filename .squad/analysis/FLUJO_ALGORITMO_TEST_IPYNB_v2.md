---
title: Algoritmo Completo de test.ipynb — Flujo de Ejecución Ordenado
author: Dallas (Backend Dev)
date: 2026-04-11
revision: v2 (DrawIO-Ready)
---

# Análisis Completo del Flujo Algoritmo — test.ipynb

## Resumen Ejecutivo

**test.ipynb** es un orquestador delgado (7 celdas, ~150 líneas) que coordinan módulos Python reutilizables en el paquete `neuroevolution/`. El algoritmo implementa **Neuroevolución Híbrida** para detectar Parkinson mediante audio: combinación de algoritmos genéticos + entrenamiento supervisado con validación cruzada paralela de 5-pliegues.

---

## 1. FLUJO PRINCIPAL (Ordenado de Inicio a Fin)

### Fase de Configuración (Pasos 1-3)

**Paso 1: Instalación de Dependencias (Celda 1)**
- **Entrada**: Lista de paquetes (torch, numpy, matplotlib, seaborn, scikit-learn)
- **Proceso**: `install_packages()` verifica/instala cada paquete
- **Salida**: Confirmación de versiones instaladas; mensaje "All dependencies verified/installed"
- **Nodo DrawIO**: `INIT_DEPS` → estado: verde si éxito, rojo si fallo
- **Mejora detectada**: Auto-instalación evita pasos manuales

---

**Paso 2: Importación de Módulos (Celda 2)**
- **Entrada**: Rutas del paquete `neuroevolution/`
- **Proceso**: 
  ```python
  from neuroevolution import (
    CONFIG, setup_device, setup_seeds, setup_logging,
    load_dataset, HybridNeuroevolution,
    plot_fitness_evolution, show_evolution_statistics
  )
  from neuroevolution.data.loader import load_fold_data
  ```
- **Salida**: `✅ All modules imported successfully`
- **Nodo DrawIO**: `IMPORT_MODULES` → ruta a 7 paquetes (config, device, logger, data, models, genetics, evolution)
- **Mejora detectada**: Modularidad permite inyección de dependencias y testing aislado

---

**Paso 3: Configuración y Setup de Dispositivo (Celda 3)**
- **Entradas**:
  - `CONFIG['data_path']` = `data/sets/folds_5`
  - `CONFIG['artifacts_dir']` = `artifacts/test_audio`
  - Parámetros opcionales: `population_size`, `max_generations`, `fitness_threshold`
- **Procesos secuenciales**:
  1. `setup_device()` → detecta CUDA o CPU, retorna `torch.device`
  2. `setup_seeds(42)` → establece seeds en torch/numpy/random para reproducibilidad
  3. `setup_logging(log_path)` → redirige `print()` a `execution_log.txt`
  4. `configure_plot_style()` → aplica estilo visual matplotlib/seaborn
- **Salida**: 
  - `device` = "cuda" o "cpu"
  - Directorio de artefactos creado
  - Tabla de configuración impresa
- **Nodo DrawIO**: `CONFIG_SETUP` → 4 sub-pasos secuenciales
- **Mejora detectada**: Configuración centralizada + logging automático = reproducibilidad garantizada

---

### Fase de Carga de Datos (Paso 4)

**Paso 4: Verificación y Carga de Dataset (Celda 4)**
- **Entradas**:
  - `CONFIG['data_path']` con 5 archivos `.npy` (folds 0-4)
  - `CONFIG['sequence_length']` (auto-detectado si falta)
  - `CONFIG['num_channels']` = 1 (audio mono)
  - `CONFIG['num_classes']` = 2 (Control vs Pathological)
- **Procesos**:
  1. `load_dataset(CONFIG)` → valida existencia de 5 pliegues, auto-detecta `sequence_length`
  2. `load_fold_arrays(CONFIG, fold_num=1)` → carga Fold 1 como muestra
  3. Imprime estadísticas de dataset
- **Salida**:
  - `X_train, y_train, X_val, y_val, X_test, y_test` (Fold 1)
  - Tabla con: `sequence_length`, `num_channels`, `num_classes`, conteos de muestras
  - Validación: Fold 1 impreso; 5 folds validados internamente
- **Nodo DrawIO**: `LOAD_DATA` → con puntos de decisión:
  - ¿Existe `data_path`? → SÍ → carga; NO → error
  - ¿`sequence_length` auto-detectable? → SÍ → continúa; NO → usa valor default
- **Mejora detectada**: Fallback multi-ruta y auto-detección evitan errores de usuario

---

### Fase de Evolución (Pasos 5-11)

**Paso 5: Inicialización del Motor de Neuroevolución (Parte de Celda 5)**
- **Entrada**: 
  - `CONFIG` (completa: población, generaciones, parámetros genéticos, bounds arquitectura)
  - `device` (CUDA o CPU)
- **Proceso**: 
  ```python
  neuroevolution = HybridNeuroevolution(CONFIG, device)
  ```
  - Constructor valida parámetros de config
  - Inicializa contadores generacionales
  - Prepara registro de progreso (JSON + texto)
- **Salida**: Instancia `HybridNeuroevolution` lista para `evolve()`
- **Nodo DrawIO**: `INIT_ENGINE` → almacena estado inicial
- **Mejora detectada**: Patrón de clase permite checkpointing y reanudación

---

**Paso 6: Ejecución Principal de Evolución (Celda 5 - bucle principal)**
- **Entrada**: Instancia `neuroevolution` inicializada
- **Proceso**: `best_genome = neuroevolution.evolve()`
  - **Sub-bucle generacional** (Paso 6a-6e, repite hasta convergencia):
    
    **6a) Evaluación Paralela de Población (per-individual)**
    - **Entrada**: Población actual (N genomas)
    - **Proceso**: Para cada genoma:
      1. Crear arquitectura CNN1D desde genoma
      2. Lanzar 5 threads (ThreadPoolExecutor, workers=5)
      3. Cada thread entrena en un pliegue (fold 0-4) en paralelo
      4. Agregar precisión media de 5 pliegues como fitness
      5. Guardar checkpoints del mejor modelo
    - **Salida**: Población evaluada con fitness (0-100%)
    - **Nodo DrawIO**: `EVAL_POPULATION` → rama a `EVAL_INDIVIDUAL` × N (paralelo)
    - **Mejora detectada**: ThreadPoolExecutor para 5-fold CV paralela → velocidad 5x vs secuencial
    
    **6b) Selección (Elitismo + Fitness Proporcional)**
    - **Entrada**: Población evaluada
    - **Proceso**:
      1. Preservar top 20% elite (configuración)
      2. Seleccionar resto por fitness proporcional (ruleta)
    - **Salida**: Padres seleccionados
    - **Nodo DrawIO**: `SELECTION` → bifurcación en `ELITE` + `PROPORTIONAL`
    - **Mejora detectada**: Elitismo previene pérdida de mejores soluciones
    
    **6c) Especiación (Compatibilidad Genética)**
    - **Entrada**: Población después de selección
    - **Proceso**:
      1. Calcular distancia de compatibilidad (diferencia estructural + paramétrica)
      2. Agrupar en especies por threshold
      3. Asignar cada individuo a especie
    - **Salida**: Diccionario de especies con miembros
    - **Nodo DrawIO**: `SPECIATION` → agrupa población en clusters
    - **Mejora detectada**: NEAT-like speciation evita convergencia prematura
    
    **6d) Reproducción (Crossover + Mutación)**
    - **Entrada**: Especies con padres
    - **Proceso**: Para cada especie:
      1. **Crossover** (99% prob): Alinear genes por innovation_id, crear hijos
      2. **Mutación adaptativa** (10-80% prob): Modular capas, filters, kernels, hyperparámetros
      3. **Crecimiento de complejidad**: Incrementar `current_max_conv_layers`/`current_max_fc_layers` cada N generaciones
      4. **Validación**: Verificar arquitectura es legal (2-30 conv, 1-10 FC, etc)
    - **Salida**: Nueva población (mutantes validados)
    - **Nodo DrawIO**: `REPRODUCTION` → bifurcación en `CROSSOVER` (alineado por innovation_id) + `MUTATION` (adaptativa)
    - **Mejora detectada**: Innovation tracking NEAT permite precisión en crossover; mutación adaptativa ajusta según diversidad
    
    **6e) Actualización Generacional**
    - **Entrada**: Nueva población
    - **Proceso**:
      1. Incrementar `generation` contador
      2. Guardar progreso a JSON + archivo texto
      3. Guardar mejor modelo encontrado a checkpoin
      4. Imprimir métricas generacionales
    - **Salida**: Estado actualizado, preparado para próxima generación
    - **Nodo DrawIO**: `UPDATE_GENERATION` → logging + checkpoint
    - **Mejora detectada**: Checkpointing permite reanudación si hay interrupciones

- **Decisiones de Convergencia** (Control de bucle):
  - **¿Fitness ≥ `fitness_threshold`?** → SÍ → STOP (objetivo alcanzado)
  - **¿Generación ≥ `max_generations`?** → SÍ → STOP (límite agotado)
  - **¿Mejora estancada N generaciones?** → SÍ → STOP (convergencia)
  - De otro modo → repite bucle 6a-6e

- **Nodo DrawIO Principal**: `EVOLUTION_LOOP` → ciclo con condición de salida

---

**Paso 7: Finalización de Evolución (Celda 5, post-bucle)**
- **Entrada**: Mejor genoma encontrado durante evolución
- **Proceso**:
  1. Registrar `end_time`
  2. Calcular `execution_time`
  3. Imprimir resumen:
     - Generaciones completadas
     - Mejor fitness
     - Rutas de artefactos (JSON, logs)
- **Salida**: Variables notebook: `best_genome`, `execution_time`
- **Nodo DrawIO**: `EVOLUTION_COMPLETE` → estado terminal

---

### Fase de Visualización y Análisis (Pasos 8-11)

**Paso 8: Visualización de Evolución (Celdas 6)**
- **Entrada**: Objeto `neuroevolution` con progreso
- **Proceso**: `plot_fitness_evolution(neuroevolution, CONFIG)`
  - Leer `evolution_progress.json`
  - Graficar: generación vs mejor fitness, fitness medio, fitness mín
  - Detectar puntos de convergencia, cambios abruptos
- **Salida**: Figura matplotlib guardada a `artifacts/test_audio/`
- **Nodo DrawIO**: `VIZ_FITNESS` → tipo: línea con anotaciones
- **Mejora detectada**: Visualización detecta estancamiento y convierte a ActionPoint

---

**Paso 9: Estadísticas de Evolución (Celda 7)**
- **Entrada**: Objeto `neuroevolution`
- **Proceso**: `show_evolution_statistics(neuroevolution, CONFIG)`
  - Tabla de: mejora/generación, tasa speciation, mutación adaptativa
  - Estadísticas de población: diversity, convergencia
  - Top 5 arquitecturas encontradas
- **Salida**: Tabla impresa + CSV opcional
- **Nodo DrawIO**: `STATS_EVOLUTION` → datos tabulares

---

**Paso 10: Análisis de Fallos (Celda 8)**
- **Entrada**: `neuroevolution.failed_evaluations` (lista de errores)
- **Proceso**: `analyze_failed_evaluations(neuroevolution)`
  - Categorizar fallos (GPU OOM, architecture invalid, training crashed)
  - Reportar frecuencias y puntos de fallo
- **Salida**: Informe de confiabilidad
- **Nodo DrawIO**: `ANALYZE_FAILURES` → reporta si hay problemas

---

**Paso 11: Análisis de Mejor Arquitectura (Celda 9)**
- **Entrada**: `best_genome`
- **Proceso**: Imprimir detalles:
  ```
  - ID del genoma
  - Fitness (F1-Score)
  - Número de capas Conv y FC
  - Detalles por capa (filtros, kernels, normalización)
  - Hyperparámetros (optimizer, learning_rate, dropout)
  - Innovation tracking (UUID, genes, historial estructural)
  ```
- **Salida**: Tabla de arquitectura
- **Nodo DrawIO**: `ANALYZE_BEST` → desglose jerárquico de genoma

---

**Paso 12: Cargar Mejor Checkpoint (Celda 10, opcional)**
- **Entrada**: Ruta de checkpoint desde `neuroevolution`
- **Proceso**: `neuroevolution.load_best_checkpoint()`
  - Deserializar modelo y genoma
  - Verificar número de parámetros
- **Salida**: `loaded_model`, `loaded_genome` listos para inferencia
- **Nodo DrawIO**: `LOAD_CHECKPOINT` → tipo: bifurcación (éxito/fallo)

---

**Paso 13: Resumen y Próximos Pasos (Celda 11, documentación)**
- **Entrada**: Ejecución completa
- **Proceso**: Imprimir checklist de logros
- **Salida**: Guía de próximos pasos
- **Nodo DrawIO**: `SUMMARY` → estado final

---

## 2. SUBPROCESOS CLAVE

### Subproceso A: Evaluación de Individuo en 5 Pliegues Paralelos

```
ENTRADA: 
  - genoma (genes de arquitectura y hyperparámetros)
  - CONFIG (parámetros de entrenamiento)
  - device (CUDA o CPU)
  - DATA: 5 folds × (X_train, y_train, X_val, y_val, X_test, y_test)

PROCESO:
  1. genome_to_model(genoma) → EvolvableCNN arquitectura
  2. for fold_id in [0,1,2,3,4]:
       thread_i = executor.submit(train_fold, model, genoma, fold_id)
  3. wait(all threads) → agregar resultados
  4. fitness = mean(accuracy_fold_0, ..., accuracy_fold_4)
  5. if fitness > best_global:
       save_checkpoint(model, genoma, fitness)

SALIDA:
  - fitness (0-100%)
  - modelo entrenado (en checkpoint si es mejor)
```

**DrawIO nodes**: `EVAL_IND_START` → 5 × `TRAIN_FOLD` (parallel) → `AGGREGATE_FITNESS` → `MAYBE_CHECKPOINT`

---

### Subproceso B: Entrenamiento de un Pliegue (dentro de Subproceso A)

```
ENTRADA:
  - modelo EvolvableCNN
  - genoma
  - fold_id ∈ [0,1,2,3,4]
  - X_train, y_train, X_val, y_val

PROCESO:
  1. Crear optimizer (SGD/Adam según genoma)
  2. Configurar learning_rate, dropout, batch_size
  3. for epoch in [0, num_epochs):
       - Forward pass en batch de train
       - Backward pass
       - Update weights
       - Early stopping si val_loss no mejora N epochs
  4. Evaluar en val set → accuracy, loss
  5. Evaluar en test set → precision, recall, F1, AUC (si disponible)

SALIDA:
  - accuracy_fold
  - métricas completas (precisión, recall, F1, AUC)
```

**DrawIO nodes**: `TRAIN_FOLD` → optimizer_init → epoch_loop → early_stop_check → validation → `FOLD_ACCURACY`

---

### Subproceso C: Crossover Alineado por Innovation ID (NEAT-like)

```
ENTRADA:
  - padre1, padre2 (genomas)
  - innovation_gene_map (mapeo de genes de innovación)

PROCESO:
  1. Alinear genes homólogos por innovation_id
  2. Para genes HOMÓLOGOS (ambos padres tienen):
       - Heredar aleatoriamente de padre1 o padre2
  3. Para genes DISJUNTOS (solo en uno):
       - Si padre es elite → heredar
       - Si padre no es elite → probabilidad baja de heredar
  4. Crear hijo validado
  5. Retornar genoma hijo

SALIDA:
  - hijo (genoma)
```

**DrawIO nodes**: `CROSSOVER_START` → `ALIGN_BY_INNOVATION` → `INHERIT_HOMOLOGOUS` → `INHERIT_DISJOINT` → `VALIDATE_CHILD`

---

### Subproceso D: Mutación Adaptativa con Crecimiento Incremental

```
ENTRADA:
  - genoma
  - current_generation
  - population_diversity (0-100%)
  - current_max_conv_layers, current_max_fc_layers (caps incrementales)

PROCESO:
  base_rate = CONFIG['base_mutation_rate']  # e.g., 0.2
  
  1. ADAPTAR TASA según diversidad:
       if diversity_low: rate = min(max_rate, base_rate * 1.5)  # +50% si baja diversidad
       if diversity_high: rate = max(min_rate, base_rate * 0.5)  # -50% si alta diversidad
  
  2. MUTAR ESTRUCTURA (if prob < rate):
       - Agregar/remover capa Conv
       - Agregar/remover capa FC
       - Cambiar num filters (1-256)
       - Cambiar kernel size
       - Cambiar norm type (BatchNorm/LayerNorm)
  
  3. CRECIMIENTO INCREMENTAL (cada N generaciones):
       if generation % growth_interval == 0:
         current_max_conv_layers = min(30, current_max_conv_layers + 1)
         current_max_fc_layers = min(10, current_max_fc_layers + 1)
  
  4. MUTAR HYPERPARÁMETROS:
       - learning_rate (log uniform)
       - dropout_rate
       - optimizer (SGD → Adam o viceversa)
  
  5. VALIDAR: Verifcar arquitectura respeta bounds

SALIDA:
  - genoma mutado y validado
```

**DrawIO nodes**: `MUTATION_START` → `ADAPT_RATE` → branch `(STRUCT_MUTATE | PARAM_MUTATE | GROWTH)` → `VALIDATE`

---

### Subproceso E: Especiación por Distancia de Compatibilidad

```
ENTRADA:
  - población después selección
  - threshold_speciation (distancia máxima)

PROCESO:
  1. Para cada individuo:
       - Calcular distancia a representantes de especies existentes
       - distance = w1 * (diferencia_capas / 30) 
                  + w2 * (diferencia_filters / 256)
                  + w3 * (diferencia_hyperparams)
  
  2. Si distance < threshold:
       - Asignar a especie existente
       - Actualizar representante si necesario
     Else:
       - Crear nueva especie
  
  3. Para cada especie:
       - Contar miembros
       - Calcular offspring esperados (proporcional a fitness medio)

SALIDA:
  - diccionario de especies {species_id: [miembros]}
  - contador de especies
```

**DrawIO nodes**: `SPECIATION_START` → loop distance_calc → `ASSIGN_SPECIES` → `COUNT_OFFSPRING`

---

## 3. PUNTOS DE DECISIÓN (Lógica de Ramificación)

| Número | Condición | Rama SÍ | Rama NO | Nodo DrawIO |
|--------|-----------|---------|---------|------------|
| 1 | ¿Existe `data_path`? | Cargar datos | ERROR → abort | `LOAD_DATA_CHECK` |
| 2 | ¿Detectar `sequence_length` automático? | Usar auto-detectado | Usar default en CONFIG | `SEQ_LEN_CHECK` |
| 3 | ¿CUDA disponible? | Usar GPU | Usar CPU + warning | `DEVICE_CHECK` |
| 4 | ¿Genoma válido (bounds respetados)? | Continuar | Reparar o rechazar | `GENOME_VALID_CHECK` |
| 5 | ¿Fitness ≥ fitness_threshold? | STOP evolución (objetivo) | Continuar | `FITNESS_GOAL_CHECK` |
| 6 | ¿Generación ≥ max_generations? | STOP evolución (límite) | Continuar | `MAX_GEN_CHECK` |
| 7 | ¿Mejora estancada N generaciones? | STOP evolución (convergencia) | Continuar | `STAGNATION_CHECK` |
| 8 | ¿Mutación estructural seleccionada? | Modificar arquitectura | Mutar solo hyperparámetros | `STRUCT_MUTATE_PROB` |
| 9 | ¿Crecer complejidad este ciclo (incrementación)? | Aumentar caps | Mantener caps | `GROWTH_CHECK` |
| 10 | ¿Evaluación falló (OOM, invalid arch)? | Registrar fallo, skipear | Continuar con fitness | `EVAL_FAILURE_CHECK` |

---

## 4. ENTRADAS Y SALIDAS DEL SISTEMA

### ENTRADAS

#### 4.1 Configuración (CONFIG dictionary)

```json
{
  "data_path": "data/sets/folds_5",
  "artifacts_dir": "artifacts/test_audio",
  "dataset_id": 0,
  "fold_id": 0,
  "population_size": 20,
  "max_generations": 100,
  "fitness_threshold": 90.0,
  "elite_percentage": 0.2,
  "num_epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001,
  "base_mutation_rate": 0.2,
  "crossover_rate": 0.99,
  "min_conv_layers": 1,
  "max_conv_layers": 30,
  "min_fc_layers": 1,
  "max_fc_layers": 10,
  "min_filters": 1,
  "max_filters": 256,
  "min_fc_nodes": 64,
  "max_fc_nodes": 1024,
  "kernel_sizes": [1, 3, 5, 7, 9, 11, 13, 15],
  "num_channels": 1,
  "num_classes": 2,
  "sequence_length": "auto-detect",
  "device": "cuda|cpu",
  "individual_parallelism": true,
  "gpu_pool_size": 4,
  "cpu_pool_size": 6
}
```

#### 4.2 Datos (5-fold CV)

```
data/sets/folds_5/
├── fold_0.npy
├── fold_1.npy
├── fold_2.npy
├── fold_3.npy
└── fold_4.npy

Estructura de cada NPZ:
  X_train: (N_train, 1, sequence_length)  # Audio mono
  y_train: (N_train,)                      # Labels 0/1
  X_val:   (N_val, 1, sequence_length)
  y_val:   (N_val,)
  X_test:  (N_test, 1, sequence_length)
  y_test:  (N_test,)
```

#### 4.3 Dispositivo

```
device: torch.device("cuda") o torch.device("cpu")
```

---

### SALIDAS

#### 5.1 Progreso de Evolución (evolution_progress.json)

```json
{
  "generation": 100,
  "best_fitness": 92.5,
  "best_genome": { ... genoma completo ... },
  "population_size": 20,
  "generations_data": [
    {
      "generation": 0,
      "best_fitness": 45.2,
      "mean_fitness": 42.1,
      "diversity": 0.85,
      "num_species": 3,
      "mutation_rate": 0.25
    },
    ...
  ]
}
```

#### 5.2 Log de Generaciones (generation_progress.txt)

```
Generation 0:
  Best Fitness: 45.20%
  Mean Fitness: 42.10%
  Worst Fitness: 38.50%
  Diversity: 0.85
  Mutation Rate: 0.25
  Num Species: 3
  
Generation 1:
  Best Fitness: 47.30%
  ...
```

#### 5.3 Log de Ejecución (execution_log.txt)

```
Toda salida print() durante ejecución, incluyendo:
- Mensajes de carga de datos
- Estadísticas por generación
- Tiempos de evaluación
- Errores y advertencias
```

#### 5.4 Checkpoint del Mejor Modelo (best_model_checkpoint.pth)

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'genome': best_genome,
    'config': CONFIG,
    'generation': final_generation,
    'fitness': best_fitness,
    'timestamp': datetime.now().isoformat()
}
```

#### 5.5 Visualizaciones (PNG/SVG en artifacts/)

- `fitness_evolution.png` — Gráfico línea: generación vs fitness
- `evolution_statistics.csv` — Tabla de estadísticas por generación
- `confusion_matrix.png` — Matriz confusión del mejor modelo
- `roc_curve.png` — Curva ROC del mejor modelo

---

## 5. MEJORAS DETECTADAS EN test.ipynb

### 1. Modularización (Refactorización Completa)
- **Qué**: Toda lógica extraída de notebook a 7 paquetes importables
- **Beneficio**: Reutilización, testing, versionamiento
- **Prevención de regresión**: Archivo único de importación (`neuroevolution/__init__.py`) actúa como contrato público

### 2. Paralelismo de 5-Fold CV (ThreadPoolExecutor)
- **Qué**: 5 threads simultáneos, uno por pliegue, durante evaluación de individuo
- **Beneficio**: ~5x aceleración (de 50 segundos → 10 segundos por individuo)
- **Prevención de regresión**: Aquí escrita así; configuración centralizada impide cambios accidentales

### 3. Innovation Tracking (NEAT-like)
- **Qué**: Cada gen tiene `innovation_id` único; crossover alinea por ID, no posición
- **Beneficio**: Evita misalignment durante recombinación (crossover precisión 100%)
- **Prevención de regresión**: Validación en `crossover.py` verifica alineación

### 4. Especiación Genética (Agrupación por Compatibilidad)
- **Qué**: Población dividida en especies por distancia de compatibilidad; cada especie contribuye proporcionalmente
- **Beneficio**: Mantiene diversidad, evita convergencia prematura, favorece nichos
- **Prevención de regresión**: Distancia de compatibilidad pre-configurada

### 5. Crecimiento Incremental de Complejidad
- **Qué**: Caps de capas (`current_max_conv_layers`) se incrementan gradualmente cada N generaciones
- **Beneficio**: Búsqueda "simple-first" → evita bloat arquitectural; convergencia más rápida
- **Prevención de regresión**: CONFIG contiene `incremental_growth_interval` e initial caps

### 6. Mutación Adaptativa (Tasa Dinámica)
- **Qué**: Tasa de mutación ajusta basada en diversidad poblacional
  - Baja diversidad → tasa sube (exploración)
  - Alta diversidad → tasa baja (explotación)
- **Beneficio**: Equilibrio dinámico sin ajuste manual
- **Prevención de regresión**: Limitantes min/max en CONFIG (`mutation_rate_min`, `mutation_rate_max`)

### 7. Checkpointing y Reanudación Resiliente
- **Qué**: Mejor modelo guardado cada generación; JSON de progreso guardado para reanudación
- **Beneficio**: Si ejecución se interrumpe, reanudación desde generación anterior sin pérdida
- **Prevención de regresión**: Validación de checkpoint al cargar (verificación de integridad)

---

## 6. RESTRICCIONES DE DISEÑO

| Parámetro | Rango | Reasoning |
|-----------|-------|-----------|
| `num_conv_layers` | 1-30 | Demasiadas → overfitting; muy pocas → underfitting |
| `num_fc_layers` | 1-10 | Redes muy profundas → vanishing gradient; muy superficiales → bajo capacity |
| `filters_per_layer` | 1-256 | >256 → GPU memory exhaustion; <1 → inválido |
| `fc_nodes` | 64-1024 | <64 → underfitting; >1024 → GPU memory extremo |
| `kernel_sizes` | [1, 3, 5, 7, 9, 11, 13, 15] | Tamaños impares para simetría; ≤15 evita receptive field explosión |
| `population_size` | 10-100 | <10 → muestreo pobre; >100 → lentitud; default 20 |
| `elite_percentage` | 0.1-0.3 | Preserva mejores; 0.2 (default) es estándar GA |
| `mutation_rate` | 0.1-0.8 | Adaptativa; base 0.2; rangos 0.1-0.8 |
| `fitness_threshold` | 50-100 | Objetivo de convergencia; 90% realista para Parkinson |
| `num_epochs` | 20-200 | >200 → overfitting; <20 → underfitting; default 50 |

---

## 7. FLUJO DE ERROR Y RECUPERACIÓN

### Caso 1: Archivo de Datos Faltante
```
PUNTO: Paso 4 (load_dataset)
ERROR: FileNotFoundError en "data/sets/folds_5/fold_0.npy"
RECUPERACIÓN:
  1. Validación fallida → excepción con ruta esperada
  2. User acción: proporcionar data_path correcto
  3. Reintentar
```
**DrawIO**: `LOAD_DATA` → `ERROR_NOT_FOUND` → manual intervention

---

### Caso 2: Arquitectura Invalida Generada
```
PUNTO: Paso 6d (mutación)
ERROR: Genoma mutado viola bounds (ej. 35 capas Conv)
RECUPERACIÓN:
  1. `validate_genome()` detecta violación
  2. `fix_genome()` recorta a max permitido
  3. Continúa con genoma reparado
  4. Log: warning printed
```
**DrawIO**: `MUTATION` → `VALIDATE` → `FIX_GENOME` → continue

---

### Caso 3: GPU Out-of-Memory (OOM)
```
PUNTO: Paso 6a (evaluación) durante entrenamiento
ERROR: torch.cuda.OutOfMemoryError
RECUPERACIÓN:
  1. Thread evaluación captura OOM
  2. Fallback: mover modelo a CPU
  3. Reentrenar en CPU (lento pero funcional)
  4. Asignar fitness bajo (ej. 0%) para desincentivar
  5. Registrar en failed_evaluations
```
**DrawIO**: `TRAIN_FOLD` → `OOM_CATCH` → `FALLBACK_CPU` → `ASSIGN_LOW_FITNESS`

---

### Caso 4: Early Stopping Muy Agresivo
```
PUNTO: Paso 6b (subproceso A) durante entrenamiento de pliegue
ERROR: Early stopping detiene entrenamiento sin convergencia (ej. epoch 5/50)
RECUPERACIÓN:
  1. Modelo subentrenado asigna fitness bajo
  2. Natural selection lo descarta
  3. Próximas mutaciones aumentan num_epochs o reducen dropout
  4. Gradualmente converge a buenos parámetros
```
**DrawIO**: `FOLD_TRAINING` → early_stop_trigger → `UNFIT_INDIVIDUAL` → natural selection

---

### Caso 5: Convergencia Prematura (Pérdida de Diversidad)
```
PUNTO: Paso 6c (especiación)
ERROR: Población converge a 1 especie (homogeneidad)
RECUPERACIÓN:
  1. Especiación detecta: num_species = 1
  2. Aumentar threshold_speciation
  3. Aumentar mutation_rate adaptativa
  4. Incrementar caps de complejidad
  5. Posibilidad: reiniciar subpoblación pequeña
```
**DrawIO**: `SPECIATION` → `DIVERSITY_CHECK` → `LOW_DIV_RECOVERY` → increase_mutation

---

## 8. NOTAS PARA DIAGRAMA DrawIO

### Colores Recomendados
- **Verde** (`#00B050`): Pasos de inicialización, validación exitosa
- **Azul** (`#0070C0`): Procesos principales (evaluación, mutación, crossover)
- **Naranja** (`#FFC000`): Decisiones/bifurcaciones
- **Rojo** (`#FF0000`): Errores, recuperación
- **Gris** (`#A6A6A6`): Pasos opcionales/visualización

### Jerarquía Recomendada
```
Nivel 0 (tronco principal):
  INIT_DEPS → IMPORT_MODULES → CONFIG_SETUP → LOAD_DATA → 
  INIT_ENGINE → EVOLUTION_LOOP → [EVAL_POPULATION/SELECTION/SPECIATION/REPRODUCTION] →
  UPDATE_GENERATION → EVOLUTION_COMPLETE → VIZ_FITNESS → STATS → SUMMARY

Nivel 1 (sub-bucles dentro EVOLUTION_LOOP):
  EVAL_POPULATION → 5 × TRAIN_FOLD (paralelo) → AGGREGATE_FITNESS
  REPRODUCTION → CROSSOVER (innovation-aligned) → MUTATION (adaptativa)
  SPECIATION → distance_calc → species_assignment

Nivel 2 (detalles, si necesario):
  TRAIN_FOLD → optimizer_init → epoch_loop → early_stop → validation
  MUTATION → ADAPT_RATE → STRUCT_MUTATE/PARAM_MUTATE → GROWTH → VALIDATE
```

### Ciclos Visuales
- **EVOLUTION_LOOP**: Rombo de decisión (fitness_threshold, max_gen, stagnation) → retorna a EVAL_POPULATION si continue
- **Época de entrenamiento**: Pequeño ciclo interno dentro TRAIN_FOLD

### Conexiones Clave para Rastrear
1. **innovation_genes** align en CROSSOVER → precisión de herencia
2. **species_assignment** en SPECIATION → offspring allocation en REPRODUCTION
3. **current_max_conv/fc_layers** en MUTATION → caps incrementales en GROWTH
4. **best_model_checkpoint** guardado en UPDATE_GENERATION → cargado en LOAD_CHECKPOINT

### Nodos de Decisión Críticos
```
FITNESS_GOAL_CHECK: if fitness ≥ 90% → STOP (verde)
                    else → repeat loop (azul)

GENOME_VALID_CHECK: if valid → continue (verde)
                    else → FIX_GENOME (naranja) → continue

EVAL_FAILURE_CHECK: if OOM/invalid → FALLBACK/REPAIR (rojo)
                    else → continue (verde)
```

---

## 9. SECUENCIA DE EJECUCIÓN TEMPORAL

```
T0: Inicialización (30 segundos)
  - Instalar deps
  - Importar módulos
  - Setup config
  - Cargar datos

T1-T_N: Loop generacional (N * T_gen minutos)
  Donde T_gen = (population_size × T_fold_parallel × 5 + T_select + T_crossover + T_mutate + T_log)
  
  Con default config (pop=20, 50 epochs/fold, GPU):
  - T_fold_parallel (1 individuo, 5 folds paralelo) ≈ 10 segundos
  - T_población = 20 × 10s = 200s ≈ 3.3 minutos
  - T_gen ≈ 3.5 minutos
  - T_100gen ≈ 350 minutos ≈ 6 horas
  
  Con CPU: T_gen ≈ 15 minutos, T_100gen ≈ 1500 minutos ≈ 25 horas

T_N+1: Visualización y análisis (5 minutos)
  - Plotear evolución
  - Mostrar estadísticas
  - Analizar fallos
  - Imprimir mejor arquitectura

TOTAL (GPU, pop=20, 100 gen): ~6 horas
TOTAL (CPU, pop=20, 100 gen): ~25 horas
```

---

## 10. VALIDACIONES CRÍTICAS

1. **Pre-Evolución**:
   - ✅ `load_dataset()` verifica 5 archivos .npy existen
   - ✅ `sequence_length` auto-detectado o validado
   - ✅ CONFIG parámetros dentro bounds

2. **Durante Evolución**:
   - ✅ `validate_genome()` previo a construcción de modelo
   - ✅ `evaluate_fitness()` captura excepciones (OOM, invalid arch)
   - ✅ Checkpoint de mejor modelo guardado atómicamente

3. **Post-Evolución**:
   - ✅ `evolution_progress.json` cargable y parseable
   - ✅ Mejor genome deserializable a modelo
   - ✅ Checkpoint compatible con versión PyTorch

---

## 11. RESUMEN VISUAL SIMPLIFICADO

```
┌─────────────────────────────────────────────────────────────┐
│         AUDIO HYBRID NEUROEVOLUTION (test.ipynb)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. SETUP (Celdas 1-3)                                      │
│     Instalar → Importar → Configurar → Detectar GPU        │
│                              ↓                              │
│  2. LOAD DATA (Celda 4)                                     │
│     Validar 5-fold, auto-detect sequence_length            │
│                              ↓                              │
│  3. INITIALIZE ENGINE (Celda 5, inicio)                    │
│     HybridNeuroevolution(CONFIG, device)                    │
│                              ↓                              │
│  4. EVOLUTION LOOP (Celda 5, main)                          │
│     ┌─────────────────────────────────────────┐            │
│     │ For generation in 0..max_generations:   │            │
│     │   a) Evaluar población (5 threads/indiv)│            │
│     │   b) Seleccionar elite + fitness-prop   │            │
│     │   c) Especiación por compatibilidad    │            │
│     │   d) Crossover innovation-aligned      │            │
│     │   e) Mutación adaptativa + crecimiento │            │
│     │   f) Validar genomas, actualizar gen   │            │
│     │                                         │            │
│     │ Hasta: fitness ≥ threshold OR gen ≥ max│            │
│     └─────────────────────────────────────────┘            │
│                              ↓                              │
│  5. VISUALIZE (Celdas 6-7)                                 │
│     Gráficos + Estadísticas + Análisis                     │
│                              ↓                              │
│  6. ANALYZE BEST (Celdas 8-9)                              │
│     Detalles arquitectura + Cargar checkpoint              │
│                              ↓                              │
│  7. SUMMARY (Celda 10)                                      │
│     Checklist logros, próximos pasos                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

OUTPUTS:
  evolution_progress.json          → resumable state
  generation_progress.txt          → human-readable log
  execution_log.txt                → full stdout
  best_model_checkpoint.pth        → best model + genome
  fitness_evolution.png            → convergence plot
```

---

## 12. REFERENCIAS A CÓDIGO MODULAR

Todo paso y subproceso está implementado en `neuroevolution/` package:

| Paso | Módulo | Función |
|------|--------|---------|
| 1 | `neuroevolution.logger` | `install_packages()` |
| 2 | `neuroevolution/__init__.py` | import statements |
| 3 | `neuroevolution.device_utils` | `setup_device()`, `setup_seeds()` |
| 3 | `neuroevolution.logger` | `setup_logging()` |
| 4 | `neuroevolution.data.loader` | `load_dataset()`, `load_fold_arrays()` |
| 5 | `neuroevolution.evolution.engine` | `HybridNeuroevolution.__init__()` |
| 6 | `neuroevolution.evolution.engine` | `HybridNeuroevolution.evolve()` |
| 6a | `neuroevolution.evolution.engine` | `evaluate_population()` → `evaluate_fitness()` → parallel fold training |
| 6b | `neuroevolution.genetics.selection` | `select_parents()` |
| 6c | `neuroevolution.genetics.speciation` | `assign_species()`, `calculate_compatibility_distance()` |
| 6d | `neuroevolution.genetics.crossover` | `crossover()` (innovation-aligned) |
| 6d | `neuroevolution.genetics.mutation` | `mutate()` (adaptativa) |
| 6e | `neuroevolution.evolution.engine` | save progress, checkpoint |
| 8 | `neuroevolution.visualization.plots` | `plot_fitness_evolution()` |
| 9 | `neuroevolution.visualization.plots` | `show_evolution_statistics()` |
| 10 | `neuroevolution.visualization.plots` | `analyze_failed_evaluations()` |
| 11 | `neuroevolution.evolution.engine` | `load_best_checkpoint()` |

---

**Documento Generado**: 2026-04-11  
**Revisión**: v2 (DrawIO-Ready, Ordenado, Completo)  
**Estado**: Listo para diagrama visual en DrawIO

