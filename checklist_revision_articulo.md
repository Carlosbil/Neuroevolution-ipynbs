# Checklist De Revisión Del Artículo

## 1. Protocolo Train/Validation/Test

- [ ] Revisar todas las menciones a validación cruzada, validación, test y conjunto de evaluación.
- [ ] Confirmar que cada fold tiene tres subconjuntos separados: train, validation y test.
- [ ] Verificar que el test no se usa para monitorizar entrenamiento.
- [ ] Verificar que el test no se usa para seleccionar checkpoints.
- [ ] Verificar que el test no se usa para calcular fitness evolutivo.
- [ ] Confirmar que la aptitud evolutiva se calcula solo sobre validation.
- [ ] Revisar si el mejor individuo se reentrena o se evalúa finalmente solo sobre test.

## 2. Datos Sintéticos, Reales Y Sujetos

- [ ] Revisar contradicciones sobre el uso de datos sintéticos en train/validation/test.
- [ ] Confirmar que los datos sintéticos se usan solo en entrenamiento.
- [ ] Confirmar que validation y test contienen solo datos reales.
- [ ] Verificar que la partición se hace estrictamente a nivel de sujeto.
- [ ] Comprobar si existe un manifiesto con IDs de sujeto por fold.
- [ ] Verificar que ninguna muestra sintética derivada de sujetos de validation/test aparece en entrenamiento.
- [ ] Revisar la coherencia entre Tabla 5, Sección 4.3 y el texto metodológico.

## 3. Tipo De Validación Experimental

- [ ] Revisar si el protocolo descrito es realmente validación cruzada clásica.
- [ ] Comprobar si son cinco particiones hold-out estratificadas por sujeto.
- [ ] Confirmar el esquema 60/20/20 para train/validation/test.
- [ ] Verificar si los subconjuntos de test cubren los 100 sujetos a lo largo de los folds.
- [ ] Ajustar la terminología metodológica si no corresponde a cross-validation clásica.

## 4. Métrica Para Selección De Checkpoint

- [ ] Revisar la contradicción entre Sección 3.4 y Sección 4.4.
- [ ] Confirmar si el checkpoint se selecciona por F1-score o por accuracy.
- [ ] Alinear la métrica de checkpoint con la métrica optimizada por la evolución.
- [ ] Valorar usar F1-score o una métrica balanceada como criterio principal.

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
