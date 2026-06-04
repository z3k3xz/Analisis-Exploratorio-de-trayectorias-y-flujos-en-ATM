"""
Todas las constantes de comportamiento del sistema viven aquí.
Cambiar un parámetro de procesamiento implica editar este fichero
y nada más. Los scripts importan desde aquí.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas absolutas resueltas a partir de la raíz del proyecto
# ---------------------------------------------------------------------------
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

DIR_DATOS = RAIZ_PROYECTO / "datos"
DIR_PARQUETS = DIR_DATOS / "parquet"
RUTA_METADATOS = DIR_DATOS / "flight_list.csv"

DIR_RESULTADOS = RAIZ_PROYECTO / "resultados"
DIR_PREP = DIR_RESULTADOS / "preparacion"
DIR_MACRO = DIR_RESULTADOS / "macro"
DIR_MICRO = DIR_RESULTADOS / "micro"
DIR_VALIDACIONES = DIR_RESULTADOS / "validaciones"
DIR_LOGS = DIR_RESULTADOS / "logs"

# Ficheros intermedios del pipeline de preparación
F_FUSIONADO = DIR_PREP / "datos_fusionados.parquet"
F_LIMPIO = DIR_PREP / "trayectorias_limpias.parquet"
F_PROYECTADO = DIR_PREP / "trayectorias_proyectadas.parquet"
F_NORMALIZADO = DIR_PREP / "trayectorias_normalizadas.parquet"

# Ficheros del pipeline macro
F_MATRIZ_MACRO = DIR_MACRO / "matriz_distancias_macro.npy"
F_IDS_MACRO = DIR_MACRO / "ids_vuelos_macro.npy"
F_CLUSTERS_MACRO = DIR_MACRO / "clusters_macro.parquet"
F_METRICAS_MACRO = DIR_MACRO / "metricas_macro.json"

# ---------------------------------------------------------------------------
# Ventana temporal del análisis
# ---------------------------------------------------------------------------
FECHA_INICIO = "2022-01-10"
FECHA_FIN = "2022-01-16"

# Patrón de los parquets diarios que entran al pipeline
PATRON_PARQUETS_SEMANA = [
    "2022-01-10.parquet",
    "2022-01-11.parquet",
    "2022-01-12.parquet",
    "2022-01-13.parquet",
    "2022-01-14.parquet",
    "2022-01-15.parquet",
    "2022-01-16.parquet",
]

# ---------------------------------------------------------------------------
# Columnas del dataset que entran al pipeline
# ---------------------------------------------------------------------------
COLUMNAS_TRAYECTORIAS = [
    'flight_id', 'timestamp', 'latitude', 'longitude',
    'altitude', 'groundspeed', 'vertical_rate'
]

# ---------------------------------------------------------------------------
# Parámetros de limpieza
# ---------------------------------------------------------------------------
VELOCIDAD_MAX_KT = 700      # nudos, techo de aviación comercial en GS
CAMBIO_ALT_MAX = 100        # ft/s, equivalente a 6.000 ft/min
ALTITUD_MIN = -100          # pies, margen para aeropuertos bajo nivel del mar
ALTITUD_MAX = 45000         # pies, techo operativo comercial
UMBRAL_HUECO_MAX = 300      # segundos, estándar OpenSky para continuidad
MIN_PUNTOS = 600            # puntos, ~10 min de datos a 1 pt/s
BBOX_LAT_MIN = 20.0     # grados, sur (cubre Magreb)
BBOX_LAT_MAX = 72.0     # grados, norte (cubre Spitsbergen)
BBOX_LON_MIN = -25.0    # grados, oeste (cubre Islandia/Azores)
BBOX_LON_MAX = 50.0     # grados, este (cubre Anatolia y Cáucaso)

# ---------------------------------------------------------------------------
# Proyección cartográfica
# ---------------------------------------------------------------------------
CRS_ORIGEN = "EPSG:4326"    # WGS84 lat/lon en grados
CRS_DESTINO = "EPSG:3034"   # ETRS89 / LCC Europe en metros
PIES_A_METROS = 0.3048

# ---------------------------------------------------------------------------
# Remuestreo
# ---------------------------------------------------------------------------
N_PUNTOS = 50               # puntos equidistantes por trayectoria

# ---------------------------------------------------------------------------
# Análisis macro
# ---------------------------------------------------------------------------
MIN_SAMPLES_MACRO = 10
# El usuario sigue pudiendo fijar min_cluster_size a mano vía argparse;
# si no, se calcula como el % indicado.
PCT_MIN_CLUSTER_SIZE_MACRO = 0.02   # 2% del total

# ---------------------------------------------------------------------------
# Análisis micro
# ---------------------------------------------------------------------------
RADIO_TERMINAL = 100_000    # metros, 100 km
MIN_SAMPLES_MICRO = 5
PCT_MIN_CLUSTER_SIZE_MICRO = 0.02   # 2% del total

# Aeropuertos analizados a nivel micro
AEROPUERTOS_MICRO = ["LOWW", "EKCH", "EGLL", "LEBL"]

# Solo los que tienen metering fixes declarados en validacion_metering_fixes.py
AEROPUERTOS_VALIDACION = ["LOWW", "EKCH"]

# Coordenadas oficiales de aeropuertos (WGS84), tomadas de AIPs publicados.
COORDENADAS_AEROPUERTOS = {
    # LOWW — Wien-Schwechat, ARP en AIP Austro Control
    "LOWW": {"lat": 48.110278, "lon": 16.569722, "nombre": "Wien-Schwechat"},
    # EKCH — København-Kastrup, ARP en AIP Naviair
    "EKCH": {"lat": 55.617917, "lon": 12.655972, "nombre": "København-Kastrup"},
    # LEMD — Madrid-Barajas, ARP en AIP ENAIRE
    "LEMD": {"lat": 40.471926, "lon": -3.560833, "nombre": "Madrid-Barajas"},
    # LEBL — Barcelona-El Prat, ARP en AIP ENAIRE
    "LEBL": {"lat": 41.296944, "lon": 2.078333,  "nombre": "Barcelona-El Prat"},
}

# ---------------------------------------------------------------------------
# Esquemas WED para la ablación
# ---------------------------------------------------------------------------
ESQUEMAS_DISTANCIA = ["ED", "WED"]
ESQUEMA_DEFAULT = "WED"

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEMILLA_ALEATORIA = 42

# ---------------------------------------------------------------------------
# Helpers para resolución de rutas micro
# ---------------------------------------------------------------------------
def f_micro_trayectorias(icao: str) -> Path:
    return DIR_MICRO / f"trayectorias_micro_{icao}.parquet"

def f_micro_matriz(icao: str, esquema: str = ESQUEMA_DEFAULT) -> Path:
    if esquema == ESQUEMA_DEFAULT:
        return DIR_MICRO / f"matriz_micro_{icao}.npy"
    return DIR_MICRO / f"matriz_micro_{icao}_{esquema}.npy"

def f_micro_ids(icao: str) -> Path:
    return DIR_MICRO / f"ids_micro_{icao}.npy"

def f_micro_clusters(icao: str, esquema: str = ESQUEMA_DEFAULT) -> Path:
    if esquema == ESQUEMA_DEFAULT:
        return DIR_MICRO / f"clusters_micro_{icao}.parquet"
    return DIR_MICRO / f"clusters_micro_{icao}_{esquema}.parquet"

def f_micro_metricas(icao: str, esquema: str = ESQUEMA_DEFAULT) -> Path:
    return DIR_MICRO / f"metricas_micro_{icao}_{esquema}.json"

# ---------------------------------------------------------------------------
# Crear directorios al importar el módulo (evita errores en cada script)
# ---------------------------------------------------------------------------
for d in [DIR_PREP, DIR_MACRO, DIR_MICRO, DIR_VALIDACIONES, DIR_LOGS]:
    d.mkdir(parents=True, exist_ok=True)