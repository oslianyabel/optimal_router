# Optimizador de rutas (Beta)

Aplicación web (Django + Leaflet) para calcular rutas con consumo de combustible, permitiendo:

- Inicio, múltiples destinos y estaciones de combustible definidas por el usuario.
- Cálculo de la ruta con restricciones de combustible (capacidad, consumo, combustible inicial).
- Comparativa de permutaciones ordenadas por menor combustible (y luego por distancia).
- Visualización de distancia total, tiempo estimado y combustible total.
- Overlay de carga con información del algoritmo/heurística utilizada y su complejidad.
- Herramientas de edición: agregar/eliminar estaciones y eliminar destinos.

## Requisitos

- Python 3.11+
- Dependencias Python del proyecto:
  ```bash
  pip install django requests
  ```

## Ejecución (desarrollo)

1. Posiciónate en la carpeta del proyecto (contiene `manage.py`).
2. Ejecuta el servidor de desarrollo:
   ```bash
   python manage.py runserver
   ```
3. Abre en el navegador: `http://127.0.0.1:8000/`

## Uso

- Click 1: define el punto de inicio.
- Clicks siguientes: agregan destinos.
- Botones:
  - "Agregar Estación"/"Eliminar Estación" para gestionar gasolineras.
  - "Eliminar Destino" para remover destinos individuales con click.
  - "Calcular Ruta" para obtener la ruta óptima dada la configuración de combustible.
  - "Reset":
    - Si no hay ruta dibujada pero hay marcadores, borra todo (inicio/destinos/estaciones).
    - Si hay ruta dibujada, borra únicamente la ruta y métricas, conservando marcadores.
- Panel inferior muestra tabla de puntos y, si se solicita, la comparativa de rutas.

## API

- Endpoint: `POST /api/route`
- Payload (JSON):
  ```json
  {
    "start": [lat, lon],
    "destinations": [[lat, lon], ...],
    "initial_fuel": number,
    "tank_capacity": number,
    "consumption_km_per_unit": number,
    "user_stations": [[lat, lon], ...],
    "return_to_start": boolean,
    "compare_all_orders": boolean
  }
  ```
- Respuesta: incluye GeoJSON de la ruta, métricas (distancia/tiempo/combustible), itinerario y (opcional) tabla de comparaciones por permutación.

## Algoritmo y complejidad

- Para pocos destinos se evalúan permutaciones tipo TSP (≈ O(n!)).
- Para muchos destinos se aplica la heurística del vecino más cercano (≈ O(n^2)), que no garantiza optimalidad.
- Repostaje: heurística codiciosa en función de capacidad y alcance; puede no ser globalmente óptima.

## Licencia

Este proyecto se distribuye bajo la Licencia MIT. Ver `LICENSE`.

## Derechos de autor

Copyright (c) 2025, Osliany Abel <oslianyabel@gmail.com>

Contacto: `oslianyabel@gmail.com`
