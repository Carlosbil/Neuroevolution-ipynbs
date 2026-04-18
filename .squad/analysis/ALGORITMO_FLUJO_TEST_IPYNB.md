# Diagrama de Flujo — Algoritmo Completo de test.ipynb

**Documento**: Análisis del proceso del algoritmo de neuroevolución híbrida  
**Notebook**: `test.ipynb` — Versión refactorizada con orquestación modular  
**Fecha**: 2026-04-11  
**Autor**: Dallas (Backend Dev)

---

## Flujo Principal — Secuencia Numerada

### FASE 1: CONFIGURACIÓN E INICIALIZACIÓN

**1. Instalación de dependencias**
- Verificar instalación de paquetes: PyTorch, NumPy, Matplotlib, Seaborn, scikit-learn
- Auto-instalar faltantes si es necesario
- Validar GPU CUDA si está disponible

**2. Importaciones de módulos**
- Importar desde paquete `neuroevolution`:
  - `CONFIG`, `setup_device`, `setup_seeds`, `setup_logging`
  - `load_dataset`, `HybridNeuroevolution`
  - `plot_fitness_evolution`, `show_evolution_statistics`, `analyze_failed_evaluations`, `configure_plot_style`
- Importar utilidades de datos: `load_fold_data`

**3. Configuración de dispositivo y rutas**
- Asignar ruta de datos: `data/sets/folds_5`
- Asignar ruta de artefactos: `artifacts/test_audio/`
- Setupear device (CUDA si disponible, sino CPU)
- Configurar seeds (reproducibilidad): 42
- Inicializar logging (redirección de print a archivo `execution_log.txt`)
- Configurar estilo de gráficos

---

### FASE 2: CARGA Y VALIDACIÓN DE DATOS

**4. Verificación de dataset**
- Cargar dataset desde `config['data_path']`
- Detectar automáticamente `sequence_length` de los archivos `.npy`
- Validar estructura: X_train, y_train, X_val, y_val, X_test, y_test para cada fold
- Extraer metadatos:
  - Número de canales (siempre 1 para audio mono)
  - Número de clases (2: Control vs Pathological)
  - Longitud de secuencia (tipicamente ~240,000 muestras @ 24kHz)
  - Recuentos de muestras por fold

**5. Reporte de dataset**
- Imprimir estructura verificada
- Confirmar que 5 folds están listos para evaluación paralela

---

### FASE 3: INICIALIZACIÓN DEL MOTOR DE EVOLUCIÓN

**6. Crear instancia de HybridNeuroevolution**
- Pasar: `CONFIG` dict, `device` (cuda/cpu)
- Inicializar estado interno:
  - Generation counter = 0
  - Best genome tracker = None
  - Population history = []
  - Evolution progress = {}
  - Paths para artefactos (progress.json, generation_log.txt, checkpoint)

---

### FASE 4: LOOP PRINCIPAL DE EVOLUCIÓN

**7. Generar población inicial**
- Crear `population_size` individuos (default: 20)
- Cada individuo = `genome` dict con:
  - `num_conv_layers` (1-30): cantidad de capas convolucionales
  - `conv_filters` (lista): filtros por capa (1-256)
  - `conv_kernels` (lista): tamaños de kernel (1-15)
  - `conv_norms` (lista): tipo de normalización (batch/layer/none)
  - `num_fc_layers` (1-10): cantidad de capas fully connected
  - `fc_nodes` (lista): nodos por capa (64-1024)
  - `optimizer`: tipo (Adam, SGD, RMSprop)
  - `learning_rate`: 1e-5 a 1e-2
  - `dropout_rate`: 0.2-0.6
  - `activation`: ReLU, ELU, etc.
  - NEAT enhancements: `innovation_uuid`, `innovation_genes`, `structural_history`

**8. PARA cada generación (0 hasta max_generations)**:

   **8a. Evaluación de población** [⚠️ Cuello de botella — parallelized]
   
   - Para cada individuo en población actual:
     - Crear instancia de `EvolvableCNN(genome, device)`
     - Validar arquitectura (espacios de batch norm válidos, etc.)
     - PARA cada fold (0-4) EN PARALELO [ThreadPoolExecutor 5 workers]:
       - Cargar datos del fold
       - Entrenar modelo: `num_epochs` iteraciones
       - Medir accuracy en validación
       - Guardar resultados del fold
     - Fitness = promedio de 5 fold accuracies
     - Registrar en `evolution_progress[generation]`
   
   **8b. Selección de reproducción**
   
   - Elitismo: conservar top `elite_percentage` (default 0.2 = 20%) individuos
   - Fitness-proportional selection: seleccionar restantes basados en fitness relativo
   - Individuos de elite pasan directamente a siguiente generación
   
   **8c. Reproducción: Crossover**
   
   - Para cada par de padres seleccionados:
     - NEAT-like crossover: alinear genes por innovation_id
     - Heredar genes homólogos (mismos genes estructurales)
     - Mantener genes disruptivos del padre más fit
     - Crear descendiente con genes alineados
   
   **8d. Reproducción: Mutación**
   
   - Para cada descendiente (post-crossover):
     - Aplicar mutación adaptativa:
       - Mutation rate = `base_mutation_rate` ± ajuste por diversidad poblacional
       - Mutaciones estructurales:
         - Añadir/remover capa convolucional
         - Cambiar número de filtros
         - Cambiar tamaño de kernel
       - Mutaciones parametrales:
         - Ajustar hyperparámetros (learning_rate, dropout)
         - Cambiar optimizer
       - Todas mutaciones respetan límites de `current_max_conv_layers` y `current_max_fc_layers`
       - Generar nuevo `innovation_uuid` y `innovation_genes` para tracking NEAT
   
   **8e. Crecimiento incremental de complejidad** [Feature mejorada en test.ipynb]
   
   - Si generación % 10 == 0:
     - Incrementar `current_max_conv_layers` hasta max (default 30)
     - Incrementar `current_max_fc_layers` hasta max (default 10)
     - Aplicar `incremental_growth_probability` para bias hacia arquitecturas más complejas
   
   **8f. Especiación genética** [Feature mejorada en test.ipynb]
   
   - Agrupar población en especies basadas en compatibility distance:
     - Comparar estructuras (número de capas, filtros, etc.)
     - Comparar parámetros (learning_rate, dropout)
     - Si distancia < umbral: misma especie
   - Por cada especie:
     - Calcular fitness promedio de especie
     - Aplicar selection pressure adaptativo
     - Prevenir convergencia prematura

**9. Convergencia y chequeos de parada**

   - **Condición 1**: Fitness >= `fitness_threshold` (default 90%)
   - **Condición 2**: Generaciones >= `max_generations`
   - Si se cumple cualquiera: salir del loop
   - Guardar `best_genome` en checkpoint `.pth`

---

### FASE 5: POST-EVOLUCIÓN — ANÁLISIS DE RESULTADOS

**10. Reportear mejor genoma encontrado**
- Imprimir especificaciones completas:
  - Número de capas (conv y fc)
  - Configuraciones de cada capa
  - Hyperparámetros óptimos
  - Fitness alcanzado
  - Innovation UUID (si aplica)
  - Número de eventos estructurales

**11. Visualización de progreso evolutivo**
- Graficar fitness vs. generación
- Mostrar estadísticas: mejor fitness, fitness promedio, desviación estándar
- Analizar divergencias/convergencias

**12. Análisis de evaluaciones fallidas** (si las hay)
- Reportear genomas que causaron excepciones durante evaluación
- Mostrar razones de fallo (errores de spatial dimension, etc.)

---

### FASE 6: VALIDACIÓN FINAL (Opcional)

**13. Cargar y inspeccionar mejor checkpoint**
- Recargar génoma y modelo desde archivo `.pth`
- Contar parámetros totales
- Imprimir arquitectura completa

---

## Subprocesos Clave

### SUBPROCESO A: Entrenamiento de un Individuo en un Fold

```
Input: genome, fold_data (X_train, y_train, X_val, y_val), device
1. Crear modelo EvolvableCNN(genome, device)
2. Inicializar optimizer (Adam, SGD, etc. según genome['optimizer'])
3. PARA epoch en range(num_epochs):
   a. Forward pass en X_train
   b. Calcular loss (CrossEntropyLoss)
   c. Backward pass
   d. Actualizar pesos
   e. Validar en fold_val, guardar mejor modelo (early stopping)
4. Predecir en fold_test
5. Calcular accuracy = correct_predictions / total_test_samples
6. Return: accuracy (0-1 escala, mapear a 0-100%)
Output: accuracy, trained_model
```

### SUBPROCESO B: Evaluación Paralela de Población (5-Fold CV)

```
Input: population (lista de genomas), CONFIG, device
Parallelism: ThreadPoolExecutor(max_workers=5)
1. PARA cada genoma en population:
   a. Crear lista de 5 futures (uno por fold)
   b. PARA fold in [0,1,2,3,4] EN PARALELO:
      - Ejecutar SUBPROCESO A (genome, fold_data)
      - Future.append(resultado)
   c. ESPERAR todos futures completados
   d. fitness = promedio(5 fold accuracies)
   e. Asignar fitness a genoma
2. Return: población evaluada con fitness
Output: population_with_fitness
```

### SUBPROCESO C: Mutación Adaptativa

```
Input: descendiente (genome copia), mutation_rate, current_generation
1. SI random() < mutation_rate:
   a. Elegir tipo mutación: estructural (40%) vs. paramétrica (60%)
   
   Caso A: Mutación estructural
   - random choice: [add_conv_layer, remove_conv_layer, change_filters, change_kernel_size]
   - Actualizar genome[layer_related_field]
   - Validar nuevamente contra current_max_conv_layers
   
   Caso B: Mutación paramétrica
   - random choice: [adjust_learning_rate, adjust_dropout, change_optimizer]
   - Aplicar perturbación pequeña (Gaussian ±5%)
   - Mantener rangos válidos (learning_rate: 1e-5 a 1e-2, etc.)

2. Generar nuevo innovation_uuid (NEAT tracking)
3. Crear innovation_genes alineados con mutación
4. Grabar event en structural_history
5. Return: mutated_genome
Output: descendiente_mutado
```

### SUBPROCESO D: Crossover NEAT-Like

```
Input: parent1 (fit), parent2 (less fit), crossover_rate
Output: child_genome

1. SI random() < crossover_rate:
   a. Alinear genes por innovation_id:
      - Genes homólogos: ambos padres tienen mismo gene structure
      - Genes disruptivos: solo parent1 tiene (parent1 más fit, mantener)
   
   b. Para cada gen alineado:
      - SI random() < 0.5: heredar de parent1
      - ELSE: heredar de parent2
   
   c. Para genes disruptivos (solo en parent1):
      - Mantener gene en hijo
   
   d. Generar nuevo innovation_uuid para hijo
   
2. ELSE (sin crossover): clonar parent1
3. Return: child
```

### SUBPROCESO E: Especiación

```
Input: población
Output: lista de especies (cada una es lista de genomas)

1. PARA cada genoma en población:
   a. Calcular compatibility_distance vs. representantes de species existentes
      - distance = |layers_diff| + |filters_diff| + |kernel_diff| + |params_diff|
   
   b. SI distance < threshold (default 2.0):
      - Asignar a especie existente
      - Actualizar representante de especie (promedio)
   
   c. ELSE:
      - Crear nueva especie con este genoma

2. PARA cada especie:
   a. Calcular fitness promedio
   b. Asignar quota de reproducción proporcional
   c. Seleccionar reproductores dentro de especie

3. Return: especies
```

---

## Puntos de Decisión

| # | Punto | Condición | Rama SÍ | Rama NO |
|---|-------|-----------|---------|---------|
| **D1** | ¿Existen todos 5 folds? | Archivos encontrados | Continuar evaluación | ❌ ERROR — Detener |
| **D2** | ¿Fitness >= threshold? | sí | **EXIT LOOP** → Fase 5 | Continuar generaciones |
| **D3** | ¿Gen >= max_generations? | sí | **EXIT LOOP** → Fase 5 | Continuar generaciones |
| **D4** | ¿Arquitectura válida? | no validar spatial dims | Usar modelo | ⚠️ FIJAR automático |
| **D5** | ¿Conv spatial dim > 1? | sí | Aplicar BatchNorm1D | Aplicar LayerNorm1D |
| **D6** | ¿Mutación activa? | `random() < rate` | Aplicar mutación | Pasar sin cambios |
| **D7** | ¿Crossover activo? | `random() < rate` | Combinar padres | Clonar padre mejor |
| **D8** | ¿Compatible con especie? | `distance < threshold` | Asignar a especie | Crear nueva especie |
| **D9** | ¿GPU disponible? | CUDA device exists | Usar GPU | Fallback a CPU |
| **D10** | ¿Incrementar complejidad? | `gen % 10 == 0` | Aumentar max layers | Mantener límites |

---

## Entradas y Salidas

### ENTRADAS

#### **1. Configuración (CONFIG dict)**
```python
{
  # Genética
  'population_size': 20,                    # Individuos por gen
  'max_generations': 100,                   # Max generaciones
  'fitness_threshold': 90.0,                # Fitness objetivo (%)
  'elite_percentage': 0.2,                  # Top 20% conservados
  'base_mutation_rate': 0.3,                # Prob mutación base
  'crossover_rate': 0.99,                   # Prob crossover
  
  # Espacio de búsqueda
  'min_conv_layers': 1,   'max_conv_layers': 30,
  'min_fc_layers': 1,     'max_fc_layers': 10,
  'min_filters': 1,       'max_filters': 256,
  'min_fc_nodes': 64,     'max_fc_nodes': 1024,
  
  # Entrenamiento
  'num_epochs': 50,
  'batch_size': 32,
  'learning_rate': 0.001,
  'dropout_rate': 0.3,
  
  # Datos
  'data_path': 'data/sets/folds_5',
  'dataset_id': 0,                          # ID dataset scenario
  'fold_id': 0,                             # ID fold
  'num_channels': 1,
  'num_classes': 2,
  
  # Artefactos
  'artifacts_dir': 'artifacts/test_audio',
}
```

#### **2. Datos de Audio**
```
data/sets/folds_5/{fold_id}/
├── X_train.npy  → (N_train, sequence_length) @ float32
├── y_train.npy  → (N_train,) @ int64 {0:Control, 1:Pathological}
├── X_val.npy    → (N_val, sequence_length)
├── y_val.npy    → (N_val,)
├── X_test.npy   → (N_test, sequence_length)
└── y_test.npy   → (N_test,)
```
*Típicamente: sequence_length = ~240,000 (10 sec @ 24kHz)*

#### **3. Device (Hardware)**
- GPU (CUDA): NVIDIA, AMD, etc. (si disponible)
- CPU: Siempre fallback

---

### SALIDAS

#### **1. Archivo de Progreso (JSON)**
```
artifacts/test_audio/evolution_progress.json

{
  "generation": 0,
  "population_size": 20,
  "best_fitness": 75.34,
  "avg_fitness": 68.45,
  "std_fitness": 8.12,
  "best_genome": {genome dict completo},
  "generation_details": [
    {
      "generation": 0,
      "best": 75.34,
      "avg": 68.45,
      "std": 8.12,
      "num_species": 3,
      "num_mutations": 12,
      "time_sec": 1234.5
    },
    ...
  ]
}
```

#### **2. Log de Generaciones (TXT)**
```
artifacts/test_audio/generation_progress.txt

========== GENERATION 0 ==========
Population size: 20
Best fitness: 75.34%
Average fitness: 68.45%
Std Dev: 8.12
Number of species: 3
Mutations applied: 12
Generation time: 1234.5s
Elite size: 4
...

========== GENERATION 1 ==========
...
```

#### **3. Checkpoint del Mejor Modelo (.pth)**
```
artifacts/test_audio/best_model_checkpoint.pth

{
  'genome': {genome dict completo},
  'model_state': {state_dict del modelo PyTorch},
  'config': {CONFIG dict},
  'generation': 47,
  'fitness': 94.56,
  'timestamp': '2026-04-11T15:30:22'
}
```

#### **4. Log de Ejecución (TXT)**
```
artifacts/test_audio/execution_log.txt

[Redirigido desde print() durante toda la ejecución]
- Mensajes de setup
- Mensajes de evolución (generation, fitness)
- Warnings de validación
- Estadísticas finales
```

#### **5. Gráficos de Visualización (PNG)**
```
artifacts/test_audio/
├── fitness_evolution.png     (Línea best + avg fitness)
├── evolution_statistics.png  (Histogramas, boxplots)
└── failed_evaluations.html   (Si hay fallos)
```

---

## Mejoras Implementadas Detectadas en test.ipynb

### 1. **NEAT-Like Innovation Tracking**
   - Cada genoma obtiene `innovation_uuid` único
   - Genes (capas, parámetros) tienen `innovation_id` para alineación
   - Durante crossover: genes homólogos se alinean exactamente
   - Previene: misalignment de estructura, pérdida de información

### 2. **Genetic Speciation**
   - Población se divide en especies por compatibility distance
   - Cada especie tiene umbral y representante
   - Selección dentro de especie (preserva diversidad global)
   - Previene: convergencia prematura, presión de selección excesiva

### 3. **Incremental Complexity Growth**
   - `current_max_conv_layers` comienza bajo (~5)
   - `current_max_fc_layers` comienza bajo (~3)
   - Aumentan gradualmente cada N generaciones
   - `incremental_growth_probability` bias estructuras complejas
   - Previene: bloat temprano, exploración controlada del espacio

### 4. **Adaptive Mutation Rate**
   - Base rate ajustada según diversidad poblacional
   - Baja diversidad → tasa más alta (exploración)
   - Alta diversidad → tasa más baja (explotación)
   - Rango: `base_mutation_rate` ± 30%

### 5. **Structural History Tracking**
   - Registro de eventos mutativos por genoma
   - Linaje completo: qué mutación, cuándo, dónde
   - Útil para: análisis de convergencia, debug

### 6. **Parallel 5-Fold CV**
   - Cada individuo evaluado en 5 folds EN PARALELO
   - ThreadPoolExecutor(max_workers=5)
   - Speedup ~4-5x vs. evaluación secuencial
   - Robusto contra overfitting (promedio de 5 folds)

### 7. **Modularidad Refactorizada** ✨ *Cambio arquitectural principal*
   - Notebook pasó de 3,800 líneas a 7 celdas
   - Todo algoritmo extraído a módulos importables
   - Package `neuroevolution/` con 7 sub-packages, 19+ módulos
   - Validación-first, error handling, OS-independent paths

---

## Restricciones y Límites de Diseño

| Parámetro | Rango | Razón |
|-----------|-------|-------|
| `num_conv_layers` | 1-30 | Audio largo, sin "linealidad excesiva" |
| `num_fc_layers` | 1-10 | Evitar overfitting en FC |
| `conv_filters` | 1-256 | Memoria GPU, computational cost |
| `fc_nodes` | 64-1024 | Idem |
| `kernel_size` | [1,3,5,7,9,11,13,15] | Receptive field estándar para audio |
| `dropout_rate` | 0.2-0.6 | Regularización sin perder información |
| `learning_rate` | 1e-5 to 1e-2 | SGD stability, convergence speed |
| `batch_size` | típicamente 32 | GPU memory constraint |
| `num_epochs` | típicamente 50 | Tiempo entrenamiento por fold |
| `fitness_threshold` | 85-95% | Problema moderadamente difícil (control vs. Parkinson) |

---

## Flujo de Error y Recuperación

| Error | Punto | Acción |
|-------|-------|--------|
| Fold no encontrado | D1 | ❌ Detener ejecución |
| Arquitectura inválida (spatial dim) | D5 | ⚠️ Fijar automático, continuar |
| GPU OOM durante evaluación | D9 | Fallback a CPU, aviso |
| Mutación causa error de shape | D6 | Validar genome post-mutación, revertir si falla |
| Crossover genera conflicto de genes | D7 | Usar gen disruptivo del padre fit |

---

## Notas para Diagrama DrawIO

1. **Colores recomendados:**
   - Verde: Inicialización, entradas
   - Azul: Loop principal, procesamiento
   - Naranja: Decisiones
   - Rojo: Puntos de error/salida

2. **Jerarquía visual:**
   - Nivel 1: FASE (Setup, Data, Evolution, Results)
   - Nivel 2: Paso principal (Generar población, Evaluar, etc.)
   - Nivel 3: Subproceso (Evaluar 1 individuo, Mutar gen, etc.)

3. **Ciclos:**
   - Loop de generación: arco hacia atrás desde convergence check
   - Loop de 5 folds: paralelismo (5 líneas paralelas)
   - Loop de crossover/mutación: N descendientes por generación

4. **Conexiones clave:**
   - Arquitectura generada → Validación → Entrenamiento
   - Entrenamiento (5 folds paralelos) → Agregación fitness
   - Selección → Crossover → Mutación → Siguiente generación

---

## Resumen Ejecutivo

**test.ipynb** implementa un **algoritmo genético híbrido** para evolucionar arquitecturas Conv1D optimizadas para clasificación de voz (Parkinson detection):

- **Entrada**: CONFIG, 5 folds de audio (~240K muestras cada uno)
- **Proceso**: 100+ generaciones de población de 20 individuos
  - Cada individuo evaluado en 5 folds EN PARALELO
  - Selección élite + fitness-proporcional
  - Crossover NEAT-like con alineación de genes
  - Mutación adaptativa (estructural + paramétrica)
  - Especiación por compatibility distance
  - Crecimiento incremental de complejidad
- **Salida**: 
  - Mejor genoma (~ 8-12 capas Conv, 2-5 capas FC, 94-96% fitness)
  - Checkpoint entrenado (`.pth`)
  - Progreso evolution (JSON + TXT)
  - Visualizaciones (PNG)

**Mejoras detectadas**: NEAT innovation tracking, speciation, incremental growth, parallel 5-fold CV, modularidad refactorizada.

---
