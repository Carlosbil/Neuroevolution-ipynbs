# Dataset de audio real y BigVSAN por persona

El cuaderno `old_vowels_dataset_por_persona.ipynb` genera el dataset completo en
`artifacts/old_vowels_dataset/`. La implementación compartida se encuentra en
`scripts/prepare_old_vowels_dataset.py` y también se puede ejecutar desde la raíz:

```powershell
.venv/Scripts/python.exe scripts/prepare_old_vowels_dataset.py
```

Requiere NumPy, pandas y SciPy. El cuaderno añade Matplotlib para visualizar el
reparto. Se necesita `artifacts/vowels_dataset/people.csv`, generado por
`vowels_dataset_por_persona.ipynb`: se copian sus asignaciones exactas por persona,
no se hace un nuevo sorteo. Las fuentes y el dataset original se conservan.

## Fuentes y reparto

Se utilizan exclusivamente las cuatro carpetas de `old_data_do_not_use`:

- `control_files_short_24khz`: control real.
- `pathological_files_short_24khz`: patológico real.
- `pretrained_40_1e5_BigVSAN_generated_control`: control generado.
- `pretrained_40_1e5_BigVSAN_generated_pathological`: patológico generado.

Los 300 ficheros auxiliares de macOS se excluyen porque no contienen audio WAV.
Las grabaciones corresponden a la vocal A, con tres repeticiones por persona.
Cada persona mantiene juntas sus grabaciones reales y las variantes generadas
que se conservan. La clase forma parte del identificador de persona.

| Partición | Personas (control / patológico) | Reales | Generados | Total |
|---|---:|---:|---:|---:|
| train | 35 / 35 | 210 | 8.399 | 8.609 |
| val | 5 / 5 | 30 | 1.200 | 1.230 |
| test | 10 / 10 | 60 | 0 | 60 |

**Hemos quitado los 2.399 audios generados de las personas de test**: 1.200 de
control y 1.199 patológicos. Test contiene únicamente sus 60 audios reales
(30 por clase). Los generados quitados se descartan del nuevo dataset y no se
trasladan a train ni a val, para mantener la separación por persona.
Se han quitado también sus WAV y representaciones individuales de audio y
espectrograma del directorio exportado. Los originales en `old_data_do_not_use`
se conservan. Train, val y sus cinco folds mantienen su reparto anterior.

Los cinco folds usan exclusivamente personas de train. Cada fold tiene 56
personas (28 por clase) en entrenamiento y 14 (7 por clase) en validación interna.
La validación global `val.csv` y el test global `test.csv` quedan fuera de los folds.
`distribution.csv` detalla los recuentos por partición, fold, clase y origen.

De los 11.998 generados disponibles, se conservan 9.599 en train y val; se han
quitado los 2.399 correspondientes a test. El dataset final contiene 9.899 audios
(300 reales + 9.599 generados). En las fuentes faltan estas dos variantes:

- `AVPEPUDEAC0050a2_shortened_iter11_generated.wav` (persona de train).
- `AVPEPUDEA0045a2_shortened_iter31_generated.wav` (persona de test).

## Representaciones y auditoría

Todos los WAV se copian sin cambios: 0,48 s, 24 kHz, mono PCM16 (11.520 muestras).

- `npy_audio/`: onda `float32` de forma `(1, 11520)`, escalada con PCM16 / 32768.
- `npy/`: espectrograma log-mel `float32` de forma `(64, 44)`. Se conserva la
  transformación de vowels: FFT 1024, 64 bandas, 80–8000 Hz y
  `log1p(1000 * mel_power)`. El salto se adapta a 240 muestras (10 ms) a 24 kHz.
- `arrays_audio/` y `arrays/`: arrays globales de train, val y test, y arrays de
  entrenamiento/validación interna de los cinco folds.
- `manifest.csv`: procedencia, SHA-256, persona, clase, origen, iteración, split,
  fold y rutas de las dos representaciones; una fila por grabación.
- `people.csv`: persona, split y fold; una fila por persona.
- `source_audit.json`: variantes ausentes, ficheros auxiliares excluidos y motivo
  y recuento de los generados quitados de test.
- `excluded_test_generated.csv`: listado de los 2.399 generados quitados, con
  sus rutas de origen, persona y asignación original a test.
- `dataset_config.json`, `audio_config.json` y `feature_config.json`: parámetros.
- `verification.json`: resultado de la verificación completa de salidas.

Los arrays siguen exactamente el orden de su CSV. Los CSV numeran folds 0–4 y
los arrays los numeran 1–5. El ID en los nombres de los arrays es `old_vowels`;
por ejemplo, `X_train_old_vowels_fold_1.npy`. Hay un único test global por
modalidad, sin copias de test por fold.

La verificación comprueba que test solo contiene audios reales, SHA-256 de todos
los WAV, igualdad de todas las ondas con su PCM, ausencia de personas compartidas
y correspondencia de cada fila de
cada array con su fichero individual y sus etiquetas. Se puede repetir con:

```powershell
.venv/Scripts/python.exe scripts/prepare_old_vowels_dataset.py --verify-only
```

## Uso con el entrenamiento

El dataset activo de `config.py` es `DEFAULT_DATASET = 'old_vowels'`.
`DEFAULT_INPUT_MODALITY` selecciona `'audio'` o `'spectrogram'`; el perfil aplica
automáticamente estas rutas, dimensiones e IDs a `test.ipynb` y sus scripts.
Para volver al dataset recortado, seleccionar `DEFAULT_DATASET = 'vowels_half'`.
Reiniciar el kernel del notebook después de cambiar cualquiera de los selectores.

| Campo | Audio | Espectrograma |
|---|---|---|
| `input_modality` | `audio` | `spectrogram` |
| `fold_files_subdirectory` | `arrays_audio` | `arrays` |
| `num_channels` | 1 | 1 |
| `num_frequency_bins` | `None` | 64 |
| `sequence_length` | 11520 | 44 |

En ambos casos: `data_path = artifacts/old_vowels_dataset`,
`dataset_id = old_vowels`, `num_folds = 5`, `audio_sample_rate_hz = 24000`,
`audio_duration_s = 0.48` y
`dataset_preparation_notebook = old_vowels_dataset_por_persona.ipynb`.
El perfil establece también `evolution_dataset_id` y `final_dataset_id` a
`old_vowels`, y los directorios de ambas fases a la modalidad elegida.
La comprobación `configure_vowels_input` utiliza el ID configurado.
`test.ipynb` obtiene los recuentos de los CSV y comprueba que test solo contiene
60 audios reales. Los resultados van a `artifacts/test_old_vowels_<modalidad>/`.
El runner del ganador usa un directorio propio por dataset y modalidad, y el
script científico lee la configuración de esa ejecución guardada en `results.json`.

Este reparto garantiza la separación de personas en los archivos exportados.
Las carpetas de origen no documentan las personas usadas para entrenar BigVSAN;
esa independencia del entrenamiento del generador no se puede comprobar aquí.
