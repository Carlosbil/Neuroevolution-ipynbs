# Evaluación de la implementación frente a NEAT

## Conclusión

**No: la implementación no es NEAT clásico.** Es un algoritmo de neuroevolución híbrida para buscar arquitecturas CNN, con varios mecanismos inspirados en NEAT. La denominación precisa es **"NEAT-like"** o **neuroevolución con mecanismos inspirados en NEAT**.

La conclusión no implica que el algoritmo sea incorrecto: significa que su representación genética y su ciclo de reproducción no implementan los elementos definitorios de NEAT clásico.

## Tests ejecutados

| Comando | Resultado |
|---|---:|
| `python -m pytest tests/test_neat_like_mechanisms.py -v` | 7/7 correctos |
| `python -m pytest -q` | 13/13 correctos |

Las pruebas específicas verifican identificadores deterministas, historial estructural, cruce alineado por identificadores, agrupación por especie, *fitness sharing* aislado y límites de complejidad incrementales.

## Comprobación de los elementos solicitados

| Elemento de NEAT clásico | Estado | Evidencia y evaluación |
|---|---|---|
| Genes de nodos y conexiones | **Ausente** | El genoma codifica parámetros de una CNN (`filters`, `kernel_sizes`, `fc_nodes`, activaciones e hiperparámetros) en `neuroevolution/genetics/genome.py`. `innovation.py` crea genes para valores de capas, no nodos ni conexiones con extremos, peso, sesgo y estado habilitado/deshabilitado. |
| Marcadores históricos | **Parcial / no equivalentes** | Hay UUIDs deterministas y `structural_history` en `neuroevolution/genetics/innovation.py`. Sin embargo, el identificador depende del tipo, índice y **valor** actual del parámetro. NEAT asigna un número de innovación global a una nueva conexión estructural y lo conserva para identificar el mismo evento histórico entre linajes. |
| Especiación | **Parcial** | `neuroevolution/genetics/speciation.py` agrupa por distancia de topología, coincidencia de UUIDs e hiperparámetros. Las pruebas confirman la agrupación y el cálculo de `adjusted_fitness`. Pero `HybridNeuroevolution.selection_and_reproduction()` realiza selección por torneo sobre toda la población; no invoca la especiación ni usa el `adjusted_fitness` para asignar descendencia o seleccionar dentro de especies. Por tanto, la especiación no protege innovaciones dentro del ciclo evolutivo real. |
| Crecimiento incremental | **Presente, pero distinto** | `neuroevolution/evolution/engine.py` aumenta límites máximos de capas por generación, y `mutation.py` puede alterar el número de capas. Es una programación de complejidad de CNN, no la mutación NEAT de añadir nodos y conexiones a un grafo neuronal. |

## Diferencias decisivas con NEAT clásico

- No existe un grafo neuronal codificado por genes de nodo y de conexión.
- No existen mutaciones canónicas de "añadir conexión" ni "dividir conexión para añadir nodo".
- Los genes no conservan un marcador histórico estructural global de conexión; se regeneran a partir de los valores actuales del genoma.
- El cruce no alinea genes de conexión homólogos de un grafo: combina parámetros de capas CNN.
- La especiación está implementada y probada como utilidad, pero no participa en la selección y reproducción del motor.

## Alcance de la evidencia

El cambio que añadió estas pruebas declara explícitamente como fuera de alcance "implementar genes de nodos/conexiones clásicos" y caracteriza el método como *NEAT-like* (`openspec/changes/add-neat-like-mechanism-tests/proposal.md`). Esto concuerda con el código y con los resultados de pruebas anteriores.

## Redacción recomendada

> Se empleó un algoritmo de neuroevolución híbrida para la búsqueda de arquitecturas CNN, incorporando mecanismos inspirados en NEAT —identificadores de innovación, agrupación por compatibilidad y aumento progresivo de complejidad—. No se implementó NEAT clásico basado en genes de nodos y conexiones.
