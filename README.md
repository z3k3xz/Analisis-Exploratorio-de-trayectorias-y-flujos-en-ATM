# TFG: Análisis exploratorio de trayectorias y flujos en ATM

**Autor:** Yungu Rhee
**Tutor:** Juan A. Fdez del Pozo
**Departamento:** Inteligencia Artificial, ETSIINF (UPM)

## Descripción

Pipeline de análisis de trayectorias aéreas ADS-B sobre el espacio aéreo europeo.
A partir de datos brutos del PRC Data Challenge 2024, el sistema limpia, proyecta,
remuestrea y agrupa las trayectorias en corredores aéreos mediante clustering
HDBSCAN a dos niveles:

- **Macro**: identificación de corredores a escala continental usando distancia euclídea (ED).
- **Micro**: identificación de flujos de llegada (STARs) en el área terminal de un
  aeropuerto usando distancia euclídea ponderada (WED), basada en el esquema
  "Weighting 1" de Corrado et al. (2020).

Aeropuertos analizados a nivel micro: se configuran en `src/config.py`. Por defecto:
**LOWW** (Wien-Schwechat), **EKCH** (København-Kastrup), **EGLL** (London-Heathrow)
y **LEBL** (Barcelona-El Prat).

## Datos

Los datos proceden del [PRC Data Challenge 2024](https://doi.org/10.4121/8cb8484b-dbe7-4750-8b87-a5b1dbc621b4)
(EUROCONTROL + OpenSky Network). No se incluyen en el repositorio por su tamaño.

Ficheros necesarios:

- `datos/parquet/*.parquet` — Trayectorias ADS-B (un fichero por día)
- `datos/flight_list.csv` — Metadatos de los vuelos

Ventana temporal analizada: **semana del 10 al 16 de enero de 2022** (7 ficheros parquet).

## Estructura del proyecto

```
TFG/
├── datos/
│   ├── parquet/
│   │   ├── 2022-01-10.parquet
│   │   └── ...
│   └── flight_list.csv
│
├── resultados/                            (generado por el pipeline)
│   ├── preparacion/
│   │   ├── datos_fusionados.parquet
│   │   ├── trayectorias_limpias.parquet
│   │   ├── trayectorias_proyectadas.parquet
│   │   └── trayectorias_normalizadas.parquet
│   ├── macro/
│   │   ├── matriz_distancias_macro.npy
│   │   ├── ids_vuelos_macro.npy
│   │   ├── clusters_macro.parquet
│   │   ├── metricas_macro.json
│   │   ├── caracterizacion_macro.parquet
│   │   └── corredores_macro.png
│   ├── micro/
│   │   ├── trayectorias_micro_{ICAO}.parquet
│   │   ├── matriz_micro_{ICAO}.npy             (esquema WED, sin sufijo)
│   │   ├── matriz_micro_{ICAO}_ED.npy          (esquema ED)
│   │   ├── ids_micro_{ICAO}.npy
│   │   ├── clusters_micro_{ICAO}.parquet       (esquema WED, sin sufijo)
│   │   ├── clusters_micro_{ICAO}_ED.parquet    (esquema ED)
│   │   ├── metricas_micro_{ICAO}_{esquema}.json
│   │   └── caracterizacion_micro_{ICAO}_{esquema}.parquet
│   ├── validaciones/
│   │   ├── metering_fixes_{ICAO}_{esquema}.parquet
│   │   ├── comparacion_ed_wed_{ICAO}.png
│   │   ├── flujos_todos_LOWW_WED.png
│   │   ├── analisis_ruido_macro.png
│   │   ├── ablacion_min_cluster_macro.json
│   │   ├── ablacion_min_cluster_micro_{ICAO}_{esquema}.json
│   │   └── ablacion_n_puntos_macro.json
│   └── logs/
│
├── src/
│   ├── __init__.py
│   ├── config.py                          Constantes globales del pipeline
│   ├── utils.py                           Funciones compartidas
│   │
│   ├── preparacion/
│   │   ├── fusionar.py                    Concatena parquets diarios, cruza metadatos
│   │   ├── limpieza.py                    6 filtros de calidad (A-F)
│   │   ├── proyeccion.py                  WGS84 → LCC EPSG:3034 (always_xy=True)
│   │   └── remuestreo_espacial.py         50 puntos equidistantes por vuelo
│   │
│   ├── macro/
│   │   ├── distancias_macro.py            Matriz de distancias euclídeas NxN
│   │   ├── clustering_macro.py            HDBSCAN + métricas (Sil/DB/CH)
│   │   ├── caracterizacion_macro.py       Top rutas, altitudes, franjas horarias
│   │   └── visualizar_macro.py            Figura PNG de clusters macro
│   │
│   ├── micro/
│   │   ├── recorte_micro.py               Recorte al TMA (100 km, ARP oficial) + remuestreo
│   │   ├── distancias_micro.py            ED o WED (--esquema ED|WED)
│   │   ├── clustering_micro.py            HDBSCAN + métricas
│   │   └── caracterizacion_micro.py       Sectores de entrada, orígenes, aerolíneas
│   │
│   ├── validaciones/
│   │   ├── validacion_metering_fixes.py   Cruce cuantitativo cluster ↔ IAF publicados
│   │   ├── comparacion_ed_wed.py          Figura side-by-side ED vs WED por aeropuerto
│   │   ├── flujos_todos_loww.py           Todas las trayectorias LOWW coloreadas por cluster
│   │   ├── diagnostico_vuelos_largos.py   Detección de trayectorias con dispersión anómala
│   │   ├── analisis_ruido_macro.py        Caracterización del ruido macro
│   │   ├── ablacion_min_cluster.py        Barrido de min_cluster_size (macro/micro)
│   │   └── ablacion_n_puntos.py           Barrido de N puntos de remuestreo (macro)
│   │
│   ├── exploracion/                       Investigación exploratoria (legacy, no usa python -m)
│   │   ├── aviones.py
│   │   ├── vuelos.py
│   │   ├── filtrado.py
│   │   ├── columnas.py
│   │   ├── altitud.py
│   │   ├── ruidos.py
│   │   └── rutas_descartadas.py
│   │
│   ├── visualizacion/                     Investigación visual (legacy, rutas obsoletas TFG v1)
│   │   ├── visualizacion_id.py
│   │   ├── visualizacion_general.py
│   │   ├── comparacion_pipeline.py
│   │   └── vuelos_filtrados.py
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── dashboard_general.py           Dash + Plotly (python -m src.dashboard.dashboard_general)
│   │
│   └── script/
│       ├── pipeline_ejecutar.bat          Pipeline principal + dashboard (lee config.AEROPUERTOS_MICRO)
│       ├── pipeline_validaciones.bat      Caracterización, ED vs WED, metering fixes (lee config.AEROPUERTOS_VALIDACION)
│       └── pipeline_ablaciones.bat        Ablaciones de min_cluster_size y N puntos (lee config.AEROPUERTOS_VALIDACION)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Requisitos previos

```
pip install -r requirements.txt
```

Python 3.12+. Todos los scripts del pipeline se ejecutan desde la **raíz del proyecto** con `python -m`.

## Configuración de aeropuertos

Los aeropuertos se configuran en `src/config.py` mediante dos listas:

- **`AEROPUERTOS_MICRO`**: aeropuertos del pipeline principal. Se procesan con
  `pipeline_ejecutar.bat` y aparecen en el dashboard. Por defecto:
  `["LOWW", "EKCH", "EGLL", "LEBL"]`.
- **`AEROPUERTOS_VALIDACION`**: subconjunto con validación completa (ED vs WED,
  metering fixes, ablaciones micro). Solo los que tienen metering fixes declarados
  en `validacion_metering_fixes.py`. Por defecto: `["LOWW", "EKCH"]`.

Para añadir un aeropuerto nuevo al pipeline:

1. Añadir su ICAO y coordenadas ARP a `COORDENADAS_AEROPUERTOS` en `config.py`.
2. Añadirlo a `AEROPUERTOS_MICRO`.
3. Si tiene metering fixes, añadirlo también a `AEROPUERTOS_VALIDACION` y declarar
   los fixes en `validacion_metering_fixes.py`.

## Ejecución rápida

### 1. Pipeline principal (para el lector)

Ejecuta todo el pipeline de principio a fin y lanza el dashboard interactivo:

```
src\script\pipeline_ejecutar.bat
```

Pasos que ejecuta: fusión → limpieza → proyección → remuestreo → distancias macro →
clustering macro → recorte micro → distancias y clustering micro WED (para cada
aeropuerto de `AEROPUERTOS_MICRO`) → dashboard.

Al terminar, abre el dashboard en `http://127.0.0.1:8050`.

### 2. Validaciones (para el autor / tribunal)

Requiere haber ejecutado `pipeline_ejecutar.bat` antes. Borra validaciones anteriores
y genera las caracterizaciones, figuras comparativas ED vs WED, validación con
metering fixes y análisis de ruido para los aeropuertos de `AEROPUERTOS_VALIDACION`:

```
src\script\pipeline_validaciones.bat
```

### 3. Ablaciones (para el autor / tribunal)

Requiere haber ejecutado `pipeline_ejecutar.bat` antes. Borra ablaciones anteriores y
ejecuta los estudios de sensibilidad sobre min_cluster_size y N puntos de remuestreo
para los aeropuertos de `AEROPUERTOS_VALIDACION`:

```
src\script\pipeline_ablaciones.bat
```

### Ejecución paso a paso

Todos los scripts del pipeline se ejecutan desde la **raíz del proyecto** con `python -m`.

#### Preparación de datos

```
python -m src.preparacion.fusionar
python -m src.preparacion.limpieza
python -m src.preparacion.proyeccion
python -m src.preparacion.remuestreo_espacial
```

#### Análisis macro

```
python -m src.macro.distancias_macro
python -m src.macro.clustering_macro
```

#### Análisis micro (LOWW y EKCH, esquema WED)

```
python -m src.micro.recorte_micro
python -m src.micro.distancias_micro LOWW --esquema WED
python -m src.micro.clustering_micro LOWW --esquema WED
python -m src.micro.distancias_micro EKCH --esquema WED
python -m src.micro.clustering_micro EKCH --esquema WED
```

#### Dashboard interactivo

```
python -m src.dashboard.dashboard_general
# Abrir http://127.0.0.1:8050
```

#### Caracterización y validación

```
python -m src.macro.caracterizacion_macro
python -m src.macro.visualizar_macro
python -m src.micro.caracterizacion_micro --esquema WED
python -m src.validaciones.validacion_metering_fixes --esquema WED
python -m src.validaciones.analisis_ruido_macro
python -m src.validaciones.comparacion_ed_wed LOWW
python -m src.validaciones.comparacion_ed_wed EKCH
python -m src.validaciones.flujos_todos_loww
python -m src.validaciones.diagnostico_vuelos_largos
```

#### Variantes ED (para comparación)

```
python -m src.micro.distancias_micro LOWW --esquema ED
python -m src.micro.clustering_micro LOWW --esquema ED
python -m src.micro.distancias_micro EKCH --esquema ED
python -m src.micro.clustering_micro EKCH --esquema ED
```

#### Ablaciones

```
python -m src.validaciones.ablacion_min_cluster --nivel macro
python -m src.validaciones.ablacion_min_cluster --nivel micro --icao LOWW --esquema WED
python -m src.validaciones.ablacion_min_cluster --nivel micro --icao EKCH --esquema WED
python -m src.validaciones.ablacion_n_puntos
```

## Flujo del pipeline

```
*.parquet + flight_list.csv
        │
        ▼
  fusionar.py ──► datos_fusionados.parquet
        │
        ▼
  limpieza.py ──► trayectorias_limpias.parquet       (6 filtros: A-F)
        │
        ▼
  proyeccion.py ──► trayectorias_proyectadas.parquet (WGS84 → LCC EPSG:3034)
        │
        ▼
  remuestreo_espacial.py ──► trayectorias_normalizadas.parquet (50 pts/vuelo)
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
   ANÁLISIS MACRO                    ANÁLISIS MICRO
   distancias_macro.py               recorte_micro.py (100 km del ARP)
   clustering_macro.py               distancias_micro.py (ED o WED)
   caracterizacion_macro.py          clustering_micro.py
   visualizar_macro.py               caracterizacion_micro.py
        │                                 │
        ▼                                 ▼
   VALIDACIONES                      VALIDACIONES
   analisis_ruido_macro.py           validacion_metering_fixes.py
   ablacion_n_puntos.py              comparacion_ed_wed.py
   ablacion_min_cluster.py           flujos_todos_loww.py
   diagnostico_vuelos_largos.py      ablacion_min_cluster.py
```

## Resultados principales (semana 10–16 enero 2022)

### Macro

| Métrica                | Valor |
| ---------------------- | ----- |
| Vuelos analizados      | 4.936 |
| Clusters identificados | 12    |
| Ruido                  | 40,5% |
| Silhouette             | 0,432 |
| Davies-Bouldin         | 0,917 |
| Calinski-Harabasz      | 2.848 |

### Micro — Comparación ED vs WED

| Aeropuerto | Esquema | Clusters | Ruido     | Silhouette | Davies-Bouldin | Calinski-Harabasz |
| ---------- | ------- | -------- | --------- | ---------- | -------------- | ----------------- |
| LOWW       | ED      | 9        | 19,4%     | 0,530      | 0,605          | 1.278             |
| **LOWW**   | **WED** | **6**    | **6,8%**  | **0,736**  | **0,377**      | **2.565**         |
| EKCH       | ED      | 15       | 12,1%     | 0,597      | 0,531          | 989               |
| **EKCH**   | **WED** | **15**   | **15,9%** | **0,659**  | **0,501**      | **3.027**         |

### Validación con metering fixes (LOWW, esquema WED)

| Cluster | Vuelos | Fix dominante | % proximidad |
| ------- | ------ | ------------- | ------------ |
| C0      | 84     | PESAT         | 85,7%        |
| C1      | 45     | BALAD         | 46,7%        |
| C2      | 64     | BALAD         | 89,1%        |
| C3      | 77     | MABOD         | 79,2%        |
| C4      | 15     | —             | no alineado  |
| C5      | 42     | NERDU         | 92,9%        |

## Dependencias

```
pandas==2.2.3
numpy==1.26.4
scipy==1.13.1
pyproj==3.6.1
hdbscan==0.8.40
scikit-learn==1.5.2
matplotlib==3.9.2
plotly==5.24.1
dash==2.18.1
pyarrow==17.0.0
```

## Notas

- Todos los scripts del pipeline, validaciones y dashboard se ejecutan desde la raíz
  del proyecto con `python -m`. Las rutas se resuelven a través de `src/config.py`.
- El pipeline principal calcula únicamente el esquema **WED** para el análisis micro.
  Las variantes **ED** se calculan en el pipeline de validaciones (necesarias para la
  comparación ED vs WED). Por convención de nombres, los ficheros del esquema WED no
  llevan sufijo de esquema (p. ej. `clusters_micro_LOWW.parquet`), mientras que los del
  esquema ED sí (`clusters_micro_LOWW_ED.parquet`); el dashboard detecta los aeropuertos
  micro buscando ficheros `clusters_micro_*.parquet`.
- Solo se implementan **dos** esquemas de distancia: `ED` (euclídea) y `WED` (euclídea
  ponderada, "Weighting 1" de Corrado et al. 2020).
- Los scripts de `src/exploracion/` y `src/visualizacion/` no siguen el esquema `python -m`
  porque no forman parte del pipeline reproducible: son herramientas de investigación
  usadas durante la fase exploratoria del desarrollo, con rutas relativas del TFG v1.
  Varios de los de `visualizacion/` apuntan a rutas que ya no existen en la estructura actual.
```