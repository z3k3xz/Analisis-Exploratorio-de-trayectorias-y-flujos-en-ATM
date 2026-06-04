
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Construcción de la matriz de vectores por trayectoria
# ---------------------------------------------------------------------------
def preparar_matriz(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    vuelos = df.pivot_table(
        index='flight_id',
        columns='point_index',
        values=['x', 'y', 'altitude'],
        aggfunc='first',
    )
    # Reordenar columnas: x_0..x_49, y_0..y_49, alt_0..alt_49
    vuelos = vuelos.reindex(columns=sorted(vuelos.columns, key=lambda c: (c[0], c[1])))
    ids_vuelos = vuelos.index.values
    matriz = vuelos.values.astype(np.float64)
    return matriz, ids_vuelos


# ---------------------------------------------------------------------------
# Pesos WED (esquema único, Weighting 1 de Corrado et al. 2020)
# ---------------------------------------------------------------------------
def calcular_pesos_wed(n_puntos: int = config.N_PUNTOS) -> np.ndarray:
    from scipy.stats import beta as beta_dist
    k = np.arange(n_puntos)
    x = k / (n_puntos - 1)
    pesos = beta_dist.pdf(x, a=2, b=6)
    return pesos / pesos.max()


def aplicar_pesos_a_matriz(matriz: np.ndarray, esquema: str,
                           n_puntos: int = config.N_PUNTOS) -> np.ndarray:
    if esquema == "ED":
        return matriz.copy()

    if esquema == "WED":
        w = calcular_pesos_wed(n_puntos)
        factor_por_punto = np.sqrt(w)
        # Cada punto aporta 3 columnas (x, y, alt). El factor se replica.
        factor_matriz = np.concatenate([factor_por_punto] * 3)
        return matriz * factor_matriz

    raise ValueError(f"Esquema desconocido: {esquema}. Esperado: 'ED' o 'WED'.")


# ---------------------------------------------------------------------------
# Remuestreo espacial (compartido entre remuestreo_espacial.py y recorte_micro.py)
# ---------------------------------------------------------------------------
def remuestrear_trayectoria(x: np.ndarray, y: np.ndarray, alt: np.ndarray,
                            n_puntos: int = config.N_PUNTOS,
                            dist_minima: float = 1.0
                            ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    dx = np.diff(x)
    dy = np.diff(y)
    dalt = np.diff(alt)
    dist_segmentos = np.sqrt(dx**2 + dy**2 + dalt**2)
    dist_acum = np.concatenate([[0.0], np.cumsum(dist_segmentos)])
    dist_total = dist_acum[-1]

    if dist_total < dist_minima:
        return None

    dist_objetivo = np.linspace(0, dist_total, n_puntos)
    x_interp = np.interp(dist_objetivo, dist_acum, x)
    y_interp = np.interp(dist_objetivo, dist_acum, y)
    alt_interp = np.interp(dist_objetivo, dist_acum, alt)
    return x_interp, y_interp, alt_interp


# ---------------------------------------------------------------------------
# Coordenadas proyectadas de aeropuertos
# ---------------------------------------------------------------------------
def coordenadas_aeropuerto_lcc(icao: str) -> tuple[float, float]:

    from pyproj import Transformer
    if icao not in config.COORDENADAS_AEROPUERTOS:
        raise KeyError(
            f"ICAO {icao} no está en config.COORDENADAS_AEROPUERTOS. "
            f"Añádelo antes de procesarlo."
        )
    lat = config.COORDENADAS_AEROPUERTOS[icao]["lat"]
    lon = config.COORDENADAS_AEROPUERTOS[icao]["lon"]
    transformer = Transformer.from_crs(config.CRS_ORIGEN, config.CRS_DESTINO,
                                       always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


# ---------------------------------------------------------------------------
# Métricas internas de clustering
# ---------------------------------------------------------------------------
def calcular_metricas_internas(dist_matrix: np.ndarray,
                               etiquetas: np.ndarray) -> dict:
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score)

    n_total = len(etiquetas)
    mascara = etiquetas >= 0
    n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    n_ruido = int((etiquetas == -1).sum())
    pct_ruido = n_ruido / n_total * 100 if n_total > 0 else 0.0

    if n_clusters > 0:
        tam = pd.Series(etiquetas[etiquetas >= 0]).value_counts()
        tam_max = int(tam.max())
        pct_tam_max = tam_max / n_total * 100
    else:
        tam_max = 0
        pct_tam_max = 0.0

    if n_clusters >= 2 and mascara.sum() >= 3:
        sil = float(silhouette_score(
            dist_matrix[np.ix_(mascara, mascara)],
            etiquetas[mascara],
            metric='precomputed',
        ))

        # Para DB y CH necesitamos vectores. Reconstruimos por MDS clásico
        # a partir de la submatriz de distancias de los puntos válidos.
        # MDS clásico es determinista y reproducible; el número de componentes
        # se fija para mantener al menos 95% de la varianza.
        try:
            from sklearn.manifold import MDS
            mds = MDS(n_components=min(10, mascara.sum() - 1),
                      dissimilarity='precomputed',
                      random_state=config.SEMILLA_ALEATORIA,
                      n_init=1, max_iter=200, normalized_stress='auto')
            coords = mds.fit_transform(dist_matrix[np.ix_(mascara, mascara)])
            db = float(davies_bouldin_score(coords, etiquetas[mascara]))
            ch = float(calinski_harabasz_score(coords, etiquetas[mascara]))
        except Exception:
            db = float('nan')
            ch = float('nan')
    else:
        sil = float('nan')
        db = float('nan')
        ch = float('nan')

    return {
        'n_clusters': n_clusters,
        'n_ruido': n_ruido,
        'pct_ruido': pct_ruido,
        'tamano_max_cluster': tam_max,
        'pct_tamano_max_cluster': pct_tam_max,
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch,
    }


def guardar_metricas(metricas: dict, ruta: Path) -> None:
    """Vuelca un dict de métricas a JSON con indentación legible."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False,
                  default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else o)


# ---------------------------------------------------------------------------
# Logging consistente
# ---------------------------------------------------------------------------
def configurar_logger(nombre: str, nivel: int = logging.INFO) -> logging.Logger:

    logger = logging.getLogger(nombre)
    logger.setLevel(nivel)

    # Evitar duplicar handlers si se llama varias veces
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    handler_consola = logging.StreamHandler(sys.stdout)
    handler_consola.setFormatter(fmt)
    logger.addHandler(handler_consola)

    ruta_log = config.DIR_LOGS / f"{nombre}.log"
    ruta_log.parent.mkdir(parents=True, exist_ok=True)
    handler_fichero = logging.FileHandler(ruta_log, mode='w', encoding='utf-8')
    handler_fichero.setFormatter(fmt)
    logger.addHandler(handler_fichero)

    return logger


@contextmanager
def cronometrar(logger: logging.Logger, etiqueta: str):
    """Context manager para medir tiempos y loggearlos."""
    t0 = time.time()
    logger.info(f"Inicio: {etiqueta}")
    try:
        yield
    finally:
        t1 = time.time()
        logger.info(f"Fin   : {etiqueta} ({t1 - t0:.2f}s)")