# Lo que falta para completar el protocolo experimental y el artículo

Este documento recoge el estado del código asociado a `test.ipynb` frente a las observaciones de la tutoría. Distingue las correcciones metodológicas ya incorporadas de las tareas pendientes de implementación, ejecución experimental y redacción.

## Estado actual del notebook

`test.ipynb` es un orquestador del paquete `neuroevolution`. Actualmente:

- Usa cinco particiones con conjuntos `train`, `validation` y `test` separados.
- Entrena los pesos únicamente con `train`.
- Calcula la aptitud evolutiva como la media del F1-score de `validation` en los cinco folds.
- Selecciona el mejor estado de cada entrenamiento y el checkpoint global mediante F1 de validación.
- Mantiene `test` fuera de la evolución y de la selección de checkpoints.
- Busca arquitecturas Conv1D e hiperparámetros mediante operadores evolutivos: selección, cruce, mutación adaptativa, crecimiento incremental, trazabilidad de innovaciones y especiación.
- Muestra la arquitectura básica del mejor genoma y las gráficas de evolución.

La configuración por defecto usa `files_syn_all_N` para entrenar y seleccionar durante todo el bucle genético. Una vez elegido el genoma, la evaluación final cambia de forma explícita a `files_real_N`: reentrena desde cero con train real, selecciona el checkpoint con validation real y mide únicamente test real.

## Correcciones metodológicas ya realizadas

La corrección más importante indicada en la tutoría ya está implementada en el código:

| Requisito | Estado actual |
| --- | --- |
| Separar train, validation y test en cada fold | Implementado |
| Entrenar solo con train | Implementado |
| Usar validation para early stopping | Implementado |
| Seleccionar checkpoint mediante validation F1 | Implementado |
| Calcular fitness evolutivo con validation F1 | Implementado |
| Reservar test para una medición final | Implementado en el módulo de evaluación final |
| Usar F1, no accuracy, para seleccionar checkpoint | Implementado |

La evaluación final está implementada en `neuroevolution/evaluation/cross_validation.py`: para cada fold crea un modelo nuevo con la arquitectura ganadora, lo entrena con train, selecciona el mejor estado en validation y lo evalúa una vez sobre test.

## Desarrollo prioritario pendiente

### ~~1. Ejecutar la evaluación final desde `test.ipynb`~~

> Implementado mediante el cambio OpenSpec `add-notebook-heldout-evaluation`; falta ejecutar el experimento completo para generar resultados.

El notebook evoluciona una arquitectura, pero no invoca la evaluación final held-out. Debe incluir, inmediatamente después de `best_genome = neuroevolution.evolve()`, una celda que ejecute `evaluate_5fold_cross_validation`.

La salida debe guardar y presentar:

- métricas de test por fold;
- matrices de confusión por fold;
- media y desviación típica de accuracy, sensibilidad, especificidad, F1 y AUC;
- resultados serializados en un archivo JSON dentro del directorio de artefactos;
- una tabla directamente reutilizable en el artículo.

Sin ejecutar esta etapa, las métricas mostradas por el notebook describen el proceso de selección sobre validation, no una evaluación final independiente de test.

### 2. Manifiesto de sujetos y auditoría de particiones

Los archivos `.npy` no contienen identificadores de sujeto. Por ello no se puede demostrar actualmente que:

- la división sea estricta a nivel de sujeto;
- un sujeto no esté en más de un subconjunto del mismo fold;
- cada sujeto aparezca en test exactamente una vez a través de las cinco particiones;
- los datos sintéticos de train no procedan de sujetos de validation o test.

Hace falta generar y versionar un manifiesto por fold, preferiblemente CSV o JSON, con al menos:

- `fold`;
- `subject_id`;
- `split` (`train`, `validation` o `test`);
- `class`;
- `sample_id` o ruta de origen;
- `is_synthetic`;
- `source_subject_id` para cada muestra sintética.

También debe añadirse una validación automática que falle si hay solapamiento de sujetos entre splits o si un dato sintético de train procede de un sujeto reservado para validation/test.

### 3. Escenarios controlados de datos sintéticos

El escenario reproducible y defendible para el artículo debe ser:

- evolución genética: train y validation sintéticos únicamente; test sintético no participa en la fitness;
- evaluación final: train, validation y test reales únicamente, tras fijar el genoma ganador.

Falta código para construir, validar y registrar este escenario. También faltan ejecuciones comparables de las siguientes condiciones:

1. Solo datos reales.
2. Datos sintéticos solo en train.
3. Sin datos sintéticos.

No deben presentarse como resultados principales los escenarios donde validation o test incorporen datos sintéticos, salvo que sean análisis exploratorios claramente etiquetados.

### 4. Baselines bajo el mismo protocolo experimental

No hay implementaciones ni ejecutores comunes para las comparaciones requeridas. Deben entrenarse y evaluarse con los mismos folds, semillas, criterio de checkpoint, datos, épocas y presupuesto computacional:

- 1D-CNN manual de referencia;
- ResNet 1D;
- InceptionTime;
- CDIL-CNN;
- random search;
- Optuna/TPE.

Los resultados de otros artículos pueden usarse como contexto, pero no como una comparación controlada con el método propuesto.

### 5. Ablaciones del algoritmo propuesto

Hace falta un runner de experimentos con configuraciones bloqueadas para aislar las contribuciones del método. Como mínimo deben evaluarse:

1. Método evolutivo completo.
2. Método sin mutación adaptativa.
3. Método sin datos sintéticos.
4. Método con solo datos reales.
5. Random search con el mismo presupuesto.
6. Optuna/TPE con el mismo presupuesto.

Si se atribuye valor a la innovación, especiación o crecimiento incremental, conviene añadir variantes sin esos componentes.

### 6. Estadística y resultados por fold

El código actual calcula media y desviación típica, pero faltan análisis para las tablas científicas:

- intervalos de confianza del 95 %;
- resultados completos por fold;
- matrices de confusión por fold;
- pruebas estadísticas pareadas entre métodos, o intervalos de confianza por sujeto cuando el manifiesto esté disponible;
- tablas reproducibles para cada escenario experimental.

Las métricas de test deben almacenarse separadamente de las métricas de validation usadas durante la evolución.

### 7. Arquitectura final y coste computacional

El notebook ya imprime parte de la arquitectura ganadora, pero no genera una ficha completa y persistente para la tabla de arquitectura. Debe guardar:

- número de bloques convolucionales;
- filtros, kernels, activación y normalización por bloque;
- pooling y dropout;
- capas densas;
- optimizador, learning rate, weight decay y scheduler;
- batch size;
- número total de parámetros;
- tiempo de entrenamiento por fold y por experimento;
- memoria máxima de GPU o RAM.

### 8. Reproducibilidad del experimento

Se deben guardar automáticamente, junto a cada ejecución:

- versión de Python, PyTorch, CUDA y cuDNN;
- GPU, memoria disponible y memoria máxima utilizada;
- semillas;
- tamaño de población, generaciones, elitismo, cruce y mutación;
- criterios de parada y early stopping;
- configuración completa en YAML o JSON inmutable;
- identificador de commit de Git, cuando sea posible.

También conviene proporcionar un script o comando único para reproducir cada experimento sin depender del estado de un notebook.

## Ajustes necesarios en la redacción del artículo

- No describir el protocolo como validación cruzada clásica si no se aporta el manifiesto que demuestre la cobertura de sujetos en test. La formulación prudente es: "cinco particiones hold-out estratificadas por sujeto" o "protocolo de cinco folds train/validation/test".
- No afirmar que el método es NEAT clásico. El código incorpora mecanismos inspirados en NEAT, pero es más preciso describirlo como neuroevolución o búsqueda evolutiva de arquitecturas e hiperparámetros Conv1D con mecanismos NEAT-like.
- Indicar de forma inequívoca que F1 de validation es el criterio de fitness y de selección de checkpoint; accuracy es solo una métrica descriptiva.
- Separar en las tablas los resultados de selección en validation de los resultados finales held-out de test.
- Eliminar tablas duplicadas, completar placeholders y no dejar tablas vacías antes de enviar el artículo.

## Orden recomendado de trabajo

1. Añadir al notebook la llamada a evaluación final held-out y generar resultados de test por fold.
2. Crear el manifiesto de sujetos y las comprobaciones automáticas de ausencia de leakage.
3. Implementar el escenario "sintéticos solo en train" y sus validaciones.
4. Implementar el runner común de baselines y ablaciones.
5. Añadir intervalos de confianza, pruebas estadísticas y exportación de tablas.
6. Registrar configuración, entorno y coste computacional por ejecución.
7. Actualizar artículo, tablas y terminología después de obtener resultados nuevos bajo el protocolo definitivo.

## Conclusión

El problema de leakage entre validation y test y la contradicción F1/accuracy están corregidos en el código base. El trabajo pendiente principal es convertir esa implementación correcta en evidencia experimental auditable: evaluación final desde el notebook, trazabilidad por sujeto, baselines y ablaciones controladas, estadística, y un registro reproducible de arquitectura y coste.
