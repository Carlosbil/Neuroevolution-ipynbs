# Checklist De Revisión Del Artículo

## 1. Protocolo Train/Validation/Test

- [x] Revisar todas las menciones a validación cruzada, validación, test y conjunto de evaluación.
- [x] Confirmar que cada fold tiene tres subconjuntos separados: train, validation y test.
- [x] Verificar que el test no se usa para monitorizar entrenamiento.
- [x] Verificar que el test no se usa para seleccionar checkpoints.
- [x] Verificar que el test no se usa para calcular fitness evolutivo.
- [x] Confirmar que la aptitud evolutiva se calcula solo sobre validation.
- [x] Revisar si el mejor individuo se reentrena o se evalúa finalmente solo sobre test.

## 2. Datos Sintéticos, Reales Y Sujetos

- [x] Revisar contradicciones sobre el uso de datos sintéticos en train/validation/test.
- [x] Confirmar que los datos sintéticos se usan solo en entrenamiento.
- [x] Confirmar que validation y test contienen solo datos reales.
- [x] Verificar que la partición se hace estrictamente a nivel de sujeto.
- [x] Comprobar si existe un manifiesto con IDs de sujeto por fold.
- [x] Verificar que ninguna muestra sintética derivada de sujetos de validation/test aparece en entrenamiento.
- [x] Revisar la coherencia entre Tabla 5, Sección 4.3 y el texto metodológico.

Notas de revisión:

- Para resultados de artículo se ha fijado `files_real_N` como configuración por defecto: train/validation/test reales y 180/60/60 muestras por fold.
- `files_real_40_1e5_N` y `files_all_real_syn_n` quedan como escenarios exploratorios: no permiten afirmar que validation/test sean solo reales ni que los sintéticos procedan solo de sujetos de entrenamiento.
- No se encontró manifiesto de IDs de sujeto por fold en `data/sets/folds_5`, `data/sets.zip` ni archivos versionados. La partición estricta por sujeto y la ausencia de derivados sintéticos de validation/test no son auditables con los `.npy` actuales.
- Actualizar Tabla 5, Sección 4.3 y metodología para no afirmar uso de sintéticos solo en entrenamiento salvo que se genere/aporte un manifiesto de sujetos.

## 3. Tipo De Validación Experimental

- [x] Revisar si el protocolo descrito es realmente validación cruzada clásica.
- [x] Comprobar si son cinco particiones hold-out estratificadas por sujeto.
- [x] Confirmar el esquema 60/20/20 para train/validation/test.
- [x] Verificar si los subconjuntos de test cubren los 100 sujetos a lo largo de los folds.
- [x] Ajustar la terminología metodológica si no corresponde a cross-validation clásica.

Notas de revisión:

- No describir como validación cruzada clásica sin manifiesto que pruebe cobertura exacta de sujetos en test.
- `files_real_N` confirma 60/20/20 por muestras y balance de clases por fold.
- Hay 100 sujetos únicos inferidos de nombres de audios reales locales, pero los `.npy` no conservan IDs; la cobertura de test de los 100 sujetos es compatible con los conteos, no estrictamente verificable.
- Usar "cinco particiones hold-out estratificadas por sujeto" solo si se aporta manifiesto; si no, usar "protocolo de cinco folds train/validation/test".

## 4. Métrica Para Selección De Checkpoint

- [x] Revisar la contradicción entre Sección 3.4 y Sección 4.4.
- [x] Confirmar si el checkpoint se selecciona por F1-score o por accuracy.
- [x] Alinear la métrica de checkpoint con la métrica optimizada por la evolución.
- [x] Valorar usar F1-score o una métrica balanceada como criterio principal.

Notas de revisión:

- La implementación actual calcula la aptitud evolutiva como media del `f1_score` de validación por fold.
- La selección de checkpoints usa `checkpoint_metric`; si no se define, cae a `fitness_metric`, y si tampoco existe cae a `f1_score`.
- Se ha hecho explícito en `CONFIG` que `fitness_metric = "f1_score"` y `checkpoint_metric = "f1_score"`, para evitar que Sección 3.4 y Sección 4.4 parezcan usar criterios distintos.
- Accuracy queda como métrica descriptiva/reportada, no como criterio principal de selección.
- Para el artículo, alinear Sección 3.4 y Sección 4.4 con esta frase: "El mejor estado/checkpoint se seleccionó maximizando el F1-score de validación, que es la misma métrica usada como fitness evolutivo".
- Mantener F1-score como criterio principal es preferible a accuracy en este contexto porque es más robusto ante posibles desbalances o costes asimétricos; si se quiere una alternativa balanceada, justificar explícitamente macro-F1, balanced accuracy o AUC y usarla de forma consistente en fitness, checkpoint y tablas.

## 5. Naturaleza Del Método Propuesto

- [ ] Revisar si el artículo afirma implementar NEAT clásico.
- [ ] Comprobar si faltan elementos de NEAT: genes de nodos/conexiones, marcadores históricos, especiación y crecimiento incremental.
- [ ] Reformular el método como búsqueda evolutiva de arquitecturas 1D-CNN e hiperparámetros si corresponde.
- [ ] Ajustar título, resumen, contribuciones y discusión para evitar una caracterización metodológica incorrecta.

## 6. Estado Del Arte Y Baselines De Búsqueda

- [ ] Ampliar la discusión con random search, Bayesian optimization, Hyperband/ASHA y Optuna.
- [ ] Incluir evolutionary NAS, efficient NAS, multi-fidelity NAS y weight sharing NAS.
- [ ] Incluir population-based training y surrogate-assisted evolutionary optimization.
- [ ] Añadir baseline de arquitectura manual base.
- [ ] Añadir random search con el mismo presupuesto.
- [ ] Añadir Optuna/TPE con el mismo presupuesto.
- [ ] Añadir comparación con el algoritmo evolutivo propuesto.
- [ ] Añadir ablación sin mutación adaptativa.
- [ ] Añadir ablación sin datos sintéticos.
- [ ] Añadir versión con datos reales solamente.

## 7. Tablas Y Resultados Incompletos

- [ ] Revisar duplicación entre Tabla 9 y Tabla 10.
- [ ] Eliminar o fusionar información duplicada.
- [ ] Completar la Tabla 11 si está vacía.
- [ ] Completar la Tabla 12 si contiene placeholders.
- [ ] Completar la Tabla 14 si está incompleta.
- [ ] Añadir matriz de confusión por fold.
- [ ] Reportar resultados por fold.
- [ ] Añadir intervalos de confianza además de media y desviación típica.
- [ ] Completar el coste computacional real.

## 8. Comparación Con Modelos De Referencia

- [ ] Revisar si ResNet, LSTM-FCN, InceptionTime y CDIL-CNN usan el mismo protocolo.
- [ ] Confirmar mismas particiones, vocal, métrica, checkpoint y conjunto de test.
- [ ] Confirmar si usan los mismos datos sintéticos y presupuesto de entrenamiento.
- [ ] Reentrenar baselines bajo el mismo protocolo experimental si es necesario.
- [ ] Incluir ResNet 1D, InceptionTime, CDIL-CNN y una 1D-CNN manual.
- [ ] Incluir random search y Optuna/TPE como baselines.
- [ ] Añadir pruebas estadísticas pareadas o intervalos de confianza por sujeto.

## 9. Arquitectura Final Encontrada

- [ ] Completar la arquitectura final encontrada en la Tabla 12.
- [ ] Incluir número de bloques convolucionales, filtros, kernels y activaciones.
- [ ] Incluir normalización, pooling, dropout y capas densas.
- [ ] Incluir optimizador, learning rate, weight decay y scheduler.
- [ ] Incluir batch size, número de parámetros, tiempo de entrenamiento y memoria máxima.

## 10. Reproducibilidad Y Configuración Experimental

- [ ] Añadir versión exacta de Python.
- [ ] Añadir versión de PyTorch, CUDA y cuDNN.
- [ ] Especificar GPU y memoria disponible/usada.
- [ ] Reportar seeds.
- [ ] Reportar número de generaciones y tamaño de población.
- [ ] Reportar tasa de elitismo, probabilidad de cruce y probabilidad de mutación.
- [ ] Reportar criterio de parada.
- [ ] Añadir repositorio o material suplementario reproducible.
- [ ] Incluir configuración YAML/JSON de los experimentos.
