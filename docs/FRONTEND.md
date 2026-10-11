# Frontend explicativo de Neuroevolution

La entrada es `docs/index.html`. Funciona directamente en un navegador, sin instalación de dependencias frontend ni servicios externos. Conserva el mapa anterior en `docs/flujo-neuroevolucion.html`.

Para servirlo desde la raíz del repositorio:

```powershell
.venv/Scripts/python.exe -m http.server 8765 --bind 127.0.0.1
```

Abrir `http://127.0.0.1:8765/docs/`.

El frontend incluye siete vistas: visión general, origen de datos, procesamiento, construcción animada de arquitecturas, evolución ilustrativa, protocolo de evaluación y fuentes del código. Permite alternar audio/Conv1D y log-mel/Conv2D, explorar tres topologías, modificar la profundidad del ejemplo y controlar reproducción, pausa, velocidad, reinicio y avance manual.

## Evidencia y actualización

`frontend-data.js` contiene un snapshot exportado de los artefactos de vocales y de los archivos fuente. Se regenera con:

```powershell
.venv/Scripts/python.exe docs/tools/export_frontend_data.py
```

El exportador verifica 100 personas y 1.500 grabaciones. Lee una grabación de Control/A del train y sus arrays preparados. Para dibujar, resume la onda en 480 intervalos min/máx y el log-mel en 162 frames, promediando grupos de cuatro. No altera archivos del dataset, modelos o resultados. El snapshot contiene valores reales de esa muestra, rutas y fragmentos del código local. Antes de compartir la web, revisar la autorización para compartir esos datos de investigación.

El recorrido documenta `vowels_dataset_por_persona.ipynb` y `test.ipynb`: partición global 70/10/20 por persona, cinco folds internos sobre train, monitorización en val global y un único test global final. Distingue ese protocolo del selector genérico sintético → real del paquete.

## Alcance de las animaciones

La construcción aplica las reglas de dimensiones del código a genomas didácticos: secuencial (pooling por capa), residual (pooling tras el bloque y proyección si cambia el canal) e Inception (ramas de igual tamaño, concatenación y pooling). Los nodos agrupan convolución, normalización y activación; los grupos indican esas operaciones en sus descripciones. Los ejemplos usan BatchNorm y ReLU, una densa oculta de 64 nodos y los pooling adaptativos del modo vocales.

La evolución es una demostración de los operadores, no una reproducción de un run registrado. Representa seis de los veinte candidatos configurados y no inventa F1, tiempos, resultados clínicos ni mejoras. No conecta a un backend, ejecuta GPU, entrena redes ni consulta progreso en directo.

El modo **Enfoque guiado**, activado por defecto, añade microzooms sobre la capa o el gen relevante, un recuadro de énfasis y una flecha con explicación breve. Las señales siguen al objetivo al desplazar o redimensionar el diagrama. Puede desactivarse en procesamiento, arquitectura o evolución; el ajuste se comparte entre las tres vistas. Al pausar se congela el acercamiento. Al cambiar de vista se cancela. Con movimiento reducido se conservan las flechas, textos y recuadros, sin zoom espacial.

Las fuentes y la configuración reflejan el momento de exportación. Regenerar el snapshot después de cambiar el código o preparar nuevos artefactos. El frontend avisa si falta el archivo del snapshot. Respeta `prefers-reduced-motion`, pausa la reproducción al cambiar de vista o pestaña y permite usar todos los controles con teclado.
