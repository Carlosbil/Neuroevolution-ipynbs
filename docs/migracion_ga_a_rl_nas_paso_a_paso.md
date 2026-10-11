# Migración de búsqueda genética a aprendizaje por refuerzo para encontrar arquitecturas

## 1. Objetivo y alcance

Sustituir **el mecanismo que propone arquitecturas** por una política de aprendizaje por refuerzo (RL). Cada episodio genera un genoma completo; los cinco folds existentes lo entrenan y devuelven una recompensa de validación. Al terminar la búsqueda se fija un único genoma y se ejecuta la evaluación final ya existente.

La primera implementación recomendada es **REINFORCE con baseline móvil**, sin compartición de pesos entre arquitecturas. Es la versión más directa para este repositorio porque la recompensa solo aparece después de entrenar la CNN y no es diferenciable respecto a las decisiones de arquitectura. El controlador se entrena con gradientes; cada CNN candidata sigue entrenándose de manera supervisada. Más adelante se puede probar PPO o un método con pesos compartidos, pero eso sería otro experimento.

**Alcance inicial recomendado:** buscar topología y tamaño de la red; fijar `optimizer`, `learning_rate` y `dropout_rate` para que la mejora mida principalmente arquitectura. La sección 12 indica cómo ampliar la política a esos hiperparámetros. Para comparar RL con el algoritmo genético (GA), ambos deben recibir exactamente el mismo espacio de búsqueda y presupuesto; si se restringen opciones para RL, hay que ejecutar también el GA con esa restricción.

Este documento describe una implementación, **no afirma que la migración ya esté hecha**.

## 2. Contratos reales del repositorio

| Pieza actual | Función | Decisión para RL |
|---|---|---|
| `neuroevolution/genetics/genome.py:create_random_genome` | Produce el diccionario de arquitectura | Sustituir como generador durante búsqueda; conservar para pruebas y baseline aleatorio |
| `neuroevolution/evolution/engine.py:HybridNeuroevolution` | Inicializa población, evalúa, selecciona élites, cruza, muta y guarda progreso | Crear un `RLArchitectureSearch` separado; no ejecutar `initialize_population`, `selection_and_reproduction`, especiación ni mutación |
| `neuroevolution/evolution/fitness.py:evaluate_fitness(genome, config, device)` | Entrena folds y retorna `(fitness, best_model, metrics)` | Reutilizar como evaluador de cada episodio |
| `neuroevolution/models/genome_validator.py` | Normaliza, valida y estima parámetros | Reutilizar para enmascarar acciones y verificar el genoma final |
| `neuroevolution/models/evolvable_cnn.py:EvolvableCNN` | Selecciona CNN 1D o 2D según `input_modality` | Reutilizar sin cambiar el formato del genoma |
| `neuroevolution/evaluation/cross_validation.py` | Reentrena y evalúa la arquitectura final con la fuente `final_evaluation` | Mantener para el flujo de folds tradicional |
| `test.ipynb` | Busca en folds internos de vocales, monitoriza `val` global y ejecuta un test global al final | Crear un notebook RL análogo; reutilizar la preparación de datos y la celda final con `evaluate_single_fold` |

`evaluate_fitness` calcula un promedio de la métrica configurada en **porcentaje, de 0 a 100**, y marca `metrics['fitness_split'] == 'validation'`. Su llamada interna a `train_fold_in_thread` pide explícitamente `eval_split='validation'`. Devuelve el modelo del fold con mejor puntuación para el checkpoint de búsqueda. La evaluación final vuelve a entrenar desde cero; no debe usar ese modelo como estimación final.

El `get_default_config()` general usa folds sintéticos en la búsqueda y folds reales al final. En cambio, `test.ipynb` llama a `configure_vowels_input`, que vincula ambas fases al dataset `vowels`; sus cinco folds pertenecen a las 70 personas de `train`. Además hay 10 personas en `val` global y 20 en el único `test` global. La búsqueda RL debe respetar el flujo concreto que se ejecute, sin mezclar ambos protocolos.

## 3. Contrato de entrada y salida de la nueva búsqueda

Interfaz propuesta:

```python
search = RLArchitectureSearch(config=search_config, device=device)
best_genome = search.search()  # dict compatible con EvolvableCNN y evaluación final
```

`best_genome` debe incluir los campos que consumen el modelo y el entrenador:

```text
num_conv_layers, num_fc_layers, filters, kernel_sizes, fc_nodes,
activations, conv_topology, residual_enabled, residual_block_size,
residual_projection, inception_enabled, inception_reduction_ratio,
inception_pool_branch, normalization_type, dropout_rate,
optimizer, learning_rate, id, fitness
```

`len(filters) == len(kernel_sizes) == num_conv_layers`, `len(fc_nodes) == num_fc_layers` y `len(activations) >= max(num_conv_layers, num_fc_layers)`. Los campos `innovation_genes`, `structural_history`, `species_id`, `pareto_rank`, `skip_next_evaluation` y `cached_from_generation` pertenecen al GA; la política no necesita generarlos. Se puede añadir `search_method='reinforce'`, `episode`, `trajectory` y `reward` como metadatos que no influyan en `EvolvableCNN`.

**Separar tres objetos:**

1. `ArchitectureSpec`: únicamente las decisiones estructurales de la política, sin ID, fitness ni métricas.
2. `genome`: conversión determinista a los campos anteriores, con hiperparámetros de entrenamiento fijados por configuración.
3. `EvaluationResult`: F1, otras métricas, número de folds válidos, parámetros, tiempo, estado y ruta de checkpoint.

Así la identidad de una arquitectura no depende de su ID ni de una recompensa previa.

## 4. Paso 0: congelar el experimento de referencia

1. Guardar la versión de código, modalidad (`audio` o `spectrogram`), semilla, archivos de folds, `fitness_metric`, `fold_selection_metric`, `num_epochs`, límites de arquitectura y coste total del run GA.
2. Registrar **número real de llamadas a `evaluate_fitness`**, arquitecturas únicas, épocas entrenadas y tiempo de GPU. `population_size * max_generations` es solo un máximo nominal: el GA reutiliza élites sin entrenarlas otra vez.
3. Acordar un presupuesto RL en **evaluaciones completas** y, si hace falta, otro en horas de GPU. Definir de antemano cuántas ejecuciones independientes se harán por método.
4. Mantener `fitness_metric='f1_score'` y `fold_selection_metric='fitness_metric'` en ambos métodos. Si se cambia el criterio de checkpoint, el resultado deja de ser directamente comparable.
5. Para `test.ipynb`, comprobar por CSV que los folds internos no contienen personas de `val` ni `test`. El `val` global puede registrarse como diagnóstico, pero no alimentar la recompensa, decidir el mejor episodio, ajustar manualmente la búsqueda ni activar early stopping. El test global se abre solo después de fijar la arquitectura.

## 5. Paso 1: definir el espacio de acciones

Crear `neuroevolution/rl/search_space.py` con:

```python
class ArchitectureSearchSpace:
    def initial_state(self, config) -> SearchState: ...
    def decision_type(self, state) -> str: ...
    def legal_actions(self, state) -> list: ...
    def apply(self, state, action) -> SearchState: ...
    def is_complete(self, state) -> bool: ...
    def to_genome(self, state, config) -> dict: ...
```

Orden determinista de decisiones para un episodio:

1. `conv_topology ∈ {sequential, residual, inception}`.
2. Si es `residual`: `residual_block_size` entre las opciones configuradas. Fijar `residual_projection='auto'`.
3. Si es `inception`: `inception_reduction_ratio` y `inception_pool_branch` entre sus opciones configuradas.
4. `num_conv_layers` y `num_fc_layers` dentro de sus límites; al inicio puede aplicarse un currículo de profundidad, pero **no cambiar la definición del espacio silenciosamente** entre runs.
5. Por cada capa convolucional: `filters[i]` y `kernel_sizes[i]`.
6. Por cada capa FC: `fc_nodes[i]`.
7. `activations[i]` para `i = 0 .. max(num_conv_layers, num_fc_layers)-1`. El modelo actual comparte esta lista entre índices de capas conv y FC.
8. Opcional en el primer experimento: `normalization_type`; si se fija, usar el mismo valor en GA y RL. Fijar `dropout_rate`, `optimizer` y `learning_rate` desde el preset de entrenamiento.
9. Emitir `STOP` solo al completar todas las decisiones obligatorias. No permitir finalizaciones parciales.

Los rangos enteros actuales de filtros y nodos FC son amplios. Para un primer controlador se puede usar una rejilla explícita, por ejemplo `filters ∈ {8,16,32,64,128,256}` y `fc_nodes ∈ {64,128,256,512,768}`, siempre dentro de `CONFIG`. **Esto reduce el espacio original**; documentarlo y aplicar idéntica rejilla al GA comparativo. Si la comparación exige exactamente el espacio actual, usar categóricas sobre todos los enteros permitidos o una codificación que cubra exactamente esos valores.

### Máscaras de acciones legales

Antes de muestrear, calcular `mask[j] = True` solo si la acción puede terminar en un genoma válido. Como mínimo:

- Respetar límites de `num_conv_layers` y `num_fc_layers`.
- Para la modalidad elegida, usar `architecture_spatial_limit(config)`, `calculate_max_safe_conv_layers(...)` y el número real de operaciones de pooling. En espectrogramas, la restricción afecta a la dimensión espacial más pequeña.
- En Inception, filtrar cantidades de canales que no satisfagan `calculate_inception_branch_channels(...)` con el valor elegido de `inception_pool_branch`.
- Si `max_model_parameters` está activo, descartar una acción cuando incluso completando el resto con las opciones mínimas se excedería el límite; volver a comprobar el conteo completo antes de entrenar.
- No permitir un estado sin acciones legales. Si ocurre, tratarlo como error de diseño de máscara y registrar el estado, no hacer una reparación aleatoria.

`validate_and_fix_genome` añade elementos aleatorios cuando las listas son demasiado cortas. Por eso `to_genome` debe crear listas completas y luego verificar que la normalización **no cambió ningún campo de arquitectura**. Si cambia, registrar `invalid_spec` y corregir el espacio de acciones. No atribuir a la política una arquitectura que ella no eligió.

## 6. Paso 2: implementar la política

Crear `neuroevolution/rl/controller.py`. Una opción sencilla es un controlador autorregresivo con `nn.Embedding` para tipo de decisión y acción previa, `nn.GRUCell` para el historial y una cabeza `nn.Linear(hidden_dim, n_actions)` por tipo de decisión. La entrada inicial es un token `START`; el estado oculto se reinicia al comienzo de cada episodio. Las cabezas comparten el estado oculto, pero cada una tiene su propio número de logits.

En cada paso:

```python
logits = policy.head(decision_type, hidden_state)
masked_logits = logits.masked_fill(~legal_mask, -torch.inf)
dist = torch.distributions.Categorical(logits=masked_logits)
action_index = dist.sample()
trajectory.log_probs.append(dist.log_prob(action_index))
trajectory.entropies.append(dist.entropy())
state = search_space.apply(state, action_values[action_index])
hidden_state = policy.advance(hidden_state, decision_type, action_index)
```

`legal_mask` debe tener al menos un `True`. Guardar en la trayectoria el tipo de decisión, acción, máscara, `log_prob`, entropía y versión de la política. **No** convertir `log_prob` a `float` antes de calcular la pérdida: se perdería el grafo de gradientes. Los logits de acciones imposibles se excluyen tanto en muestreo como en cálculo de log-probabilidad.

Prueba rápida antes de entrenar CNN: generar, por ejemplo, 1000 arquitecturas, construirlas con `EvolvableCNN` para ambas modalidades que se vayan a usar y verificar que las listas y restricciones coinciden. Esto prueba el generador, no la calidad del método RL.

## 7. Paso 3: envolver la evaluación y definir la recompensa

Crear `neuroevolution/rl/evaluator.py`. Secuencia por episodio:

```python
genome = search_space.to_genome(state, config)
assert is_genome_valid(genome, config)
signature = stable_genome_signature(genome, config)
fitness, best_fold_model, metrics = evaluate_fitness(genome, config, device)
```

Antes de llamar al entrenador, construir `EvolvableCNN(genome, config)` una vez y liberar ese modelo. Sus constructores 1D y 2D ya hacen una pasada interna con tensor ficticio para calcular la forma previa a las capas FC; no hace falta repetir un `forward` adicional sobre la entrada completa. Si falla la construcción o el límite de parámetros, marcar el episodio inválido sin consumir cinco entrenamientos. Si `metrics['n_valid_folds'] != config['num_folds']`, marcar `failed_evaluation`: el código actual puede devolver fitness cero cuando hay errores y no debe confundirse con una arquitectura válida que obtuvo F1 bajo.

Recompensa de referencia para el primer experimento:

```text
r = fitness / 100.0                 si todos los folds son válidos
r = -1.0                            si la arquitectura o evaluación falla
```

El campeón se elige por el **mayor F1 de validación**, no por el baseline de RL. Si se busca calidad con límite de tamaño, imponer `max_model_parameters` como restricción dura y registrar tiempo y parámetros. Si se quiere optimizar calidad y coste simultáneamente, declarar antes del experimento una recompensa escalar, por ejemplo `r = F1/100 - λp·log1p(params/P0) - λt·log1p(segundos/T0)`, con `λp`, `λt`, `P0` y `T0` fijos. Un ranking Pareto del GA no se traslada automáticamente a REINFORCE; sería otra definición de objetivo.

Guardar tiempo medido y `estimate_genome_parameter_count`; contrastar el estimado con `sum(p.numel() for p in model.parameters())`. Los segundos son ruidosos por carga de GPU: si entran en la recompensa, registrar hardware, concurrencia y condiciones de medida.

### Caché de evaluaciones

La clave debe incluir `stable_genome_signature`, modalidad, identificador y versión de datos, folds, métrica, selección de checkpoint, `num_epochs`, semilla/evaluación y código o versión de protocolo. Nunca incluir `id` ni episodio. Si una arquitectura vuelve a salir, usar su recompensa guardada **sin perder** el `log_prob` de la trayectoria actual: sigue siendo una muestra nueva de la política. Registrar `cache_hit=True`, pero contabilizar el coste original de la arquitectura si la recompensa penaliza tiempo. No asignar tiempo cero al acierto de caché. Persistir métricas y mantener el checkpoint del campeón en una ruta estable; una entrada de caché no aporta por sí sola un `best_fold_model` para guardarlo de nuevo.

Para un experimento reproducible, fijar cómo se asignan semillas a arquitectura y fold. Como `evaluate_fitness` puede entrenar folds en threads, no cambiar la semilla global de PyTorch dentro de cada thread. Usar generadores locales o procesos aislados; para una primera verificación se puede ejecutar con `fold_parallel_workers=1` y documentar ese ajuste en ambos métodos.

## 8. Paso 4: actualizar REINFORCE a nivel de episodio

Crear `neuroevolution/rl/engine.py`. Agrupar `B` episodios generados por **la misma versión** de la política antes de `optimizer.step()`. Evaluarlos, convertir recompensas a tensor y calcular:

```text
logp_i = suma_t log πθ(a_it | s_it, máscara_it)
H_i    = suma_t entropía_it
A_i    = r_i - b
L      = -promedio_i [stop_gradient(A_i) · logp_i]
         - β · promedio_i H_i
```

`b` es una media móvil de recompensas de **lotes anteriores**: inicializarla con el primer lote completado después de calcular su pérdida, y actualizarla luego como `b ← (1-α)b + α·mean(r_lote)`. En el primer lote se puede usar `b=0` o una constante predefinida. No incluir el propio `r_i` en su baseline antes de calcular `A_i`. `β` controla exploración por entropía; registrar su valor.

Implementación del update:

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)
optimizer.step()
baseline.update(rewards.detach())
```

No reutilizar trayectorias antiguas para nuevos pasos de REINFORCE: sus `log_prob` pertenecen a otra política. Sí se puede reutilizar el resultado de una **arquitectura repetida** al puntuar una trayectoria nueva. Con muchos fallos iniciales (`r=-1`), conviene resolver primero el generador y las máscaras; una penalización por fallos no sustituye la validación del espacio.

Pseudocódigo del bucle:

```python
best_f1 = -float('inf')
evaluations = 0
while evaluations < max_evaluations:
    trajectories = [policy.sample(search_space) for _ in range(batch_size)]
    results = [evaluator.evaluate(t.genome) for t in trajectories]
    evaluations += sum(not r.cache_hit and r.trained for r in results)
    for t, result in zip(trajectories, results):
        if result.valid and result.fitness > best_f1:
            best_f1 = result.fitness
            save_champion(result.genome, result.best_fold_model, result.metrics)
    update_policy(trajectories, results, baseline)
    save_rl_state()
return load_champion_genome()
```

Añadir un máximo de **episodios totales** además de `max_evaluations`, porque una política que repite candidatos podría consumir caché indefinidamente sin agotar las evaluaciones nuevas. El evaluador debe rechazar nuevas llamadas a `evaluate_fitness` al llegar al presupuesto y cerrar con un último lote parcial de trayectorias completadas; el pseudocódigo resume el flujo, pero sin ese control un lote podría sobrepasar `max_evaluations`. Registrar número de propuestas, únicas, inválidas, aciertos de caché y evaluaciones entrenadas.

## 9. Paso 5: progreso, checkpoints y reanudación

No escribir el progreso RL en `evolution_progress.json` ni reutilizar `HybridNeuroevolution._load_evolution_progress()`: su esquema incluye población y generación. Usar, por ejemplo, `artifacts/rl_vowels_audio/rl_state.pt`, `episode_log.jsonl` y `best_architecture.json` en un directorio distinto del GA.

Guardar tras cada lote **completo**, con escritura temporal y sustitución atómica:

- `policy.state_dict()`, `optimizer.state_dict()`, estado del baseline y paso de actualización.
- Semillas y estados RNG de `random`, NumPy, `torch` y CUDA.
- Configuración efectiva, modalidad, identificador de dataset/folds, versión del espacio de acciones, métrica y presupuesto.
- Número de episodios, evaluaciones entrenadas, firmas ya evaluadas, recompensas y ruta del checkpoint campeón.
- Genoma campeón, F1 de validación y métricas agregadas.

Al reanudar, comparar configuración y huella de datos/protocolo antes de cargar la caché. Si difieren, iniciar otro experimento. Una reanudación al límite de un lote evita repetir un gradiente; si se interrumpe a mitad de lote, registrar las evaluaciones ya terminadas y reconstruir el lote sin evaluarlas de nuevo.

El checkpoint del campeón debe conservar el mismo contrato útil que el GA: `model_state_dict`, `genome`, `fitness`, `config`, `selection_split='validation'`, `fitness_metric` y `test_evaluated=False`. Guardar además `search_method='reinforce'` y episodio. No borrar el checkpoint campeón hasta que el nuevo esté escrito correctamente.

## 10. Paso 6: integrar notebooks y evaluación final

1. Crear `rl_vowels_architecture_search.ipynb` a partir del flujo de datos de `test.ipynb`. Mantener `configure_vowels_input`, validación de personas/folds y selección de `input_modality`.
2. Cambiar la celda que crea `MonitoredVowelsEvolution(...).evolve()` por `RLArchitectureSearch(...).search()`. La política usa solo las cinco validaciones internas para recompensa.
3. Si se conserva el gráfico de `val` global, medirlo en intervalos definidos antes del run y marcarlo **solo diagnóstico**. No usar ese F1 para actualizar la política ni escoger campeón.
4. Adaptar gráficos e informes que hoy esperan `generation_stats` o `fitness_history`: en RL usar `episode`, `update`, recompensa media, mejor F1 de validación, entropía, fracción inválida, arquitecturas únicas, tiempo y parámetros. No llamar a `plot_fitness_evolution` con un objeto RL sin adaptar su contrato.
5. En el flujo de vocales, tras fijar el genoma, conservar la celda final que construye `final_train_loader`, `final_val_loader`, `final_test_loader` y llama a `evaluate_single_fold(...)`. Abrir `X_test_vowels.npy` una única vez después de la búsqueda. El F1 de test nunca entra en un nuevo update o en la elección de arquitectura.
6. En el flujo general de folds sintéticos/reales, llamar a `evaluate_5fold_cross_validation(best_genome, config, device)`, que aplica `get_data_phase_config(..., 'final_evaluation')` y reentrena con datos reales. Indicar expresamente en el informe cuál de estos dos protocolos se utilizó.
7. Exportar `RLArchitectureSearch` desde `neuroevolution/rl/__init__.py`; exportarlo también desde `neuroevolution/__init__.py` solo si se desea la misma API cómoda que `HybridNeuroevolution`.

## 11. Paso 7: pruebas de aceptación y comparación

Pruebas automáticas mínimas en `tests/test_rl_search_space.py`, `tests/test_rl_controller.py`, `tests/test_rl_engine.py` y `tests/test_rl_protocol.py`:

- `to_genome()` conserva todas las acciones seleccionadas; nunca rellena listas al azar. Las arquitecturas generadas son válidas en `audio` y `spectrogram` bajo sus configuraciones correspondientes.
- Las máscaras prohíben profundidad que agota la dimensión espacial, canales Inception inválidos y exceso de parámetros. Ninguna máscara válida está vacía.
- Una muestra devuelve un `log_prob` diferenciable; una actualización con recompensas distintas produce gradientes finitos y cambia algún parámetro de la política.
- Una recompensa repetida de caché usa el `log_prob` **nuevo** de la política; la clave distingue cambios de modalidad, datos, folds, épocas y semilla.
- Con `evaluate_fitness` simulado, un candidato con menos folds válidos recibe estado de fallo; un candidato válido con F1 mayor sustituye al campeón y queda en checkpoint recuperable.
- Interrupción/reanudación en frontera de lote conserva política, optimizador, baseline, contador y campeón; no reevalúa firmas ya terminadas.
- Un test de protocolo verifica que la búsqueda solo pide `train` y `validation`, y que el acceso a `test` ocurre únicamente después de elegir el genoma.

Verificación de integración con presupuesto pequeño: ejecutar unas pocas propuestas reales, comprobar que el genoma elegido construye `EvolvableCNN`, que `n_valid_folds == num_folds`, que la recompensa corresponde al F1 registrado y que el checkpoint se puede cargar. No usar el test final para depurar esta fase.

Comparación de investigación: mismo conjunto de candidatos permitidos, mismas particiones, métrica, épocas, semillas de entrenamiento, hardware y presupuesto **real** de evaluaciones; repetir runs con varias semillas independientes. Reportar mejor F1 de validación frente a evaluaciones y frente a tiempo de GPU, diversidad de arquitecturas, parámetros del campeón y, una vez fijadas todas las reglas, el resultado del test held-out. Incluir búsqueda aleatoria como control. Una victoria sobre GA en un solo run no demuestra superioridad estadística.

## 12. Extensiones después de la primera versión

- **Hiperparámetros aprendidos:** añadir decisiones para `dropout_rate`, `optimizer`, `learning_rate` y normalización. El genoma ya admite esos campos; actualizar firmas de caché y comparación con GA. En ese caso se está buscando arquitectura **y** configuración de entrenamiento.
- **Recompensa con coste:** activar la fórmula escalar declarada en el paso 3 o una restricción dura de parámetros/tiempo. Reportar también F1 sin penalización.
- **PPO o actor-critic:** considerar si REINFORCE presenta demasiada varianza tras medirlo. Requiere almacenar observaciones, valores y probabilidades antiguas; no se obtiene cambiando únicamente la fórmula de pérdida.
- **Pesos compartidos:** podría reducir coste, pero cambia el significado del fitness y puede alterar el ranking de arquitecturas. Validar los finalistas entrenándolos desde cero antes de comparar métodos.

## 13. Orden concreto de implementación

1. Crear `rl/search_space.py` y sus pruebas de validez para ambas modalidades.
2. Crear `rl/controller.py` y comprobar muestreo, máscaras y gradientes.
3. Crear `rl/evaluator.py` como adaptador de `evaluate_fitness`; definir fallo, recompensa y caché.
4. Crear `rl/engine.py` con lotes de trayectorias, REINFORCE, baseline y campeón.
5. Añadir checkpoint/reanudación y logs propios; verificar un ciclo interrumpido.
6. Crear el notebook RL de vocales; mantener la celda de test global aislada hasta el final.
7. Hacer un run corto de integración sin mirar test, corregir fallos y congelar configuración.
8. Ejecutar comparación GA/RL/aleatoria bajo presupuesto común y luego una sola evaluación held-out por protocolo definido.

## Referencias metodológicas

- Zoph y Le, [*Neural Architecture Search with Reinforcement Learning*](https://arxiv.org/abs/1611.01578): controlador que propone arquitecturas y recibe una recompensa de validación.
- Pham et al., [*Efficient Neural Architecture Search via Parameter Sharing*](https://arxiv.org/abs/1802.03268): extensión con compartición de pesos; útil como comparación conceptual, no incluida en la implementación inicial descrita aquí.
