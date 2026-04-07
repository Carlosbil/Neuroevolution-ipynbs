# Neuroevolution-ipynbs

Proyecto de investigacion para clasificacion de voz en deteccion de Parkinson mediante un enfoque de **neuroevolucion hibrida** sobre redes **Conv1D**. El flujo principal esta implementado en notebooks para facilitar experimentacion, trazabilidad y comparacion rapida de configuraciones.

## Descripcion general

Este repositorio contiene un pipeline para:

1. Evolucionar arquitecturas de redes neuronales para audio (GA + entrenamiento supervisado).
2. Evaluar cada individuo con **5-fold cross-validation en paralelo**.
3. Conservar checkpoints del mejor modelo global durante la evolucion.
4. Realizar evaluacion final con metricas completas (Accuracy, Precision, Recall, F1, AUC, matriz de confusion).

El objetivo es encontrar arquitecturas robustas para separar clases **Control vs Pathological** en senales de voz.

## Que hay en el proyecto

### Notebooks principales

- `best_Audio_hybrid_neuroevolution_notebook.ipynb`: notebook principal con todo el pipeline de neuroevolucion, entrenamiento, evaluacion y exportacion de resultados.
- `test.ipynb`: variante de pruebas/iteracion del flujo principal.

### Diferencia entre `best` y `test`

- `best_Audio_hybrid_neuroevolution_notebook.ipynb`:
	- Usa `info_path = artifacts/best_audio` para guardar logs, progreso y checkpoints.
	- Esta orientado como version de referencia para ejecuciones completas y resultados finales.
	- En el estado actual del repositorio no conserva salidas de error en celdas.

- `test.ipynb`:
	- Usa `info_path = artifacts/test_audio` para separar artefactos de pruebas.
	- Se usa como cuaderno de experimentacion/iteracion sobre el mismo pipeline.
	- En el estado actual incluye al menos una ejecucion interrumpida (`KeyboardInterrupt`) durante `neuroevolution.evolve()`.

En resumen: ambos comparten la base metodologica, pero `best` se usa como ruta principal de resultados y `test` como espacio de pruebas.

### Datos y recursos

La carpeta `data/` incluye recursos listos para experimentacion:

- `control_files_short_24khz/` y `pathological_files_short_24khz/`: audios base por clase.
- `csv/`: tablas de caracteristicas y/o metadata para diferentes escenarios (reales, sinteticos y generados).
- `pretrained_40_1e5_BigVSAN_generated_control/` y `pretrained_40_1e5_BigVSAN_generated_pathological/`: recursos relacionados con generacion/preentrenamiento.
- `sets/`: conjuntos preparados para entrenamiento y evaluacion.

Dentro de `data/sets/` destacan:

- `folds_5/`: archivos `.npy` para 5-fold CV (train/val/test por fold).
- `generated_together_train_40_1e5_N/`: datos sinteticos para entrenamiento.
- `test_together_N/`: conjunto de prueba real.
- `test_together_syn_1_N/`: conjunto de prueba sintetico alternativo.

## Estructura resumida

```text
Neuroevolution-ipynbs/
|- best_Audio_hybrid_neuroevolution_notebook.ipynb
|- test.ipynb
|- README.md
|- LICENSE
|- data/
|  |- csv/
|  |- sets/
|  |  |- folds_5/
|  |  |- generated_together_train_40_1e5_N/
|  |  |- test_together_N/
|  |  |- test_together_syn_1_N/
|  |- control_files_short_24khz/
|  |- pathological_files_short_24khz/
|  |- pretrained_40_1e5_BigVSAN_generated_control/
|  |- pretrained_40_1e5_BigVSAN_generated_pathological/
```

## Enfoque tecnico

El notebook principal combina:

- **Algoritmo genetico** para evolucionar hiperparametros y arquitectura.
- **Conv1D para audio** como backbone de clasificacion.
- **Mutacion adaptativa** en funcion de la diversidad de poblacion.
- **Elitismo moderado** para retener los mejores individuos.
- **Evaluacion paralela por folds** con `ThreadPoolExecutor`.
- **Checkpointing del mejor global** para reutilizacion en evaluacion final.

## Configuracion del experimento

En la seccion de configuracion del notebook (`CONFIG`) se ajustan parametros clave:

- Evolucion: `population_size`, `max_generations`, `fitness_threshold`.
- Mutacion y cruce: `base_mutation_rate`, `mutation_rate_min/max`, `crossover_rate`.
- Modelo: rangos de capas Conv/FC, filtros, dropout, kernel sizes.
- Entrenamiento: `num_epochs`, `learning_rate`, criterios de early stopping.
- Datos: `data_path`, `dataset_id`, `fold_id`, `fold_files_subdirectory`.

Configuracion observada en el notebook:

- Ruta de folds: `data/sets/folds_5`
- Dataset/fold de trabajo: variantes `40_1e5_N` y escenarios documentados para `all_real_syn_n`
- Directorio de artefactos: `artifacts/best_audio`

## Requisitos

El notebook instala/importa principalmente:

- Python 3.10+
- PyTorch
- torchvision
- numpy
- matplotlib
- seaborn
- tqdm
- jupyter
- ipywidgets
- scikit-learn

Instalacion sugerida:

```bash
pip install torch torchvision numpy matplotlib seaborn tqdm jupyter ipywidgets scikit-learn
```

## Como ejecutarlo

1. Abrir `best_Audio_hybrid_neuroevolution_notebook.ipynb` en VS Code o Jupyter.
2. Verificar que la ruta de datos en `CONFIG['data_path']` apunte a `data/sets/folds_5`.
3. Ajustar `dataset_id` y `fold_id` segun el escenario que quieras evaluar.
4. Ejecutar celdas en orden:
	- imports y entorno
	- configuracion
	- carga/verificacion de datos
	- neuroevolucion
	- evaluacion final 5-fold y metricas
5. Revisar artefactos y logs generados.

## Artefactos de salida

Durante y despues de la ejecucion se generan, entre otros:

- `artifacts/best_audio/evolution_progress.json`
- `artifacts/best_audio/generation_progress.txt`
- checkpoint del mejor modelo global (actualizado dinamicamente)
- graficas y tablas de metricas finales

## Escenarios de datos (resumen)

El proyecto contempla pruebas con:

- solo datos reales
- entrenamiento sintetico y test real
- solo sinteticos
- mezcla real + sintetico

Esto permite comparar generalizacion y robustez en contextos distintos.

## Estado del proyecto

- Desarrollo activo en notebooks por restricciones de entorno academico.
- Pipeline funcional para experimentacion y comparacion de configuraciones.
- Estructura de datos preparada para multiples variantes de entrenamiento/evaluacion.

## Licencia

Este proyecto esta publicado bajo licencia **MIT**. Consulta `LICENSE` para mas detalles.
