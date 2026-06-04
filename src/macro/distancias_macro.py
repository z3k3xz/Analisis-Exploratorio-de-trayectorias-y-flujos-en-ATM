from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from src import config
from src.utils import configurar_logger, cronometrar, preparar_matriz


def main():
    logger = configurar_logger('05_distancias_macro')
    logger.info("=" * 60)
    logger.info(" MATRIZ DE DISTANCIAS EUCLÍDEAS — MACRO")
    logger.info("=" * 60)

    with cronometrar(logger, "carga trayectorias normalizadas"):
        df = pd.read_parquet(config.F_NORMALIZADO)
    n_vuelos = df['flight_id'].nunique()
    logger.info(f"Cargados {n_vuelos:,} vuelos")

    with cronometrar(logger, "construcción matriz de vectores"):
        matriz, ids_vuelos = preparar_matriz(df)
    logger.info(f"Matriz vectorizada: {matriz.shape[0]:,} vuelos x "
                f"{matriz.shape[1]} componentes")

    n_nan = int(np.isnan(matriz).sum())
    if n_nan:
        logger.warning(f"AVISO: {n_nan} NaN en la matriz vectorizada")
    else:
        logger.info("Sin NaN en la matriz vectorizada")

    n_pares = n_vuelos * (n_vuelos - 1) // 2
    logger.info(f"Pares de vuelos a calcular: {n_pares:,}")

    with cronometrar(logger, "cálculo pdist(euclidean) + squareform"):
        dist_condensada = pdist(matriz, metric='euclidean')
        dist_matrix = squareform(dist_condensada)

    tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    logger.info(f"Distancias (metros):")
    logger.info(f"  Mín:     {tri.min():>15,.0f}")
    logger.info(f"  P05:     {np.percentile(tri, 5):>15,.0f}")
    logger.info(f"  Mediana: {np.median(tri):>15,.0f}")
    logger.info(f"  Media:   {tri.mean():>15,.0f}")
    logger.info(f"  P95:     {np.percentile(tri, 95):>15,.0f}")
    logger.info(f"  Máx:     {tri.max():>15,.0f}")

    with cronometrar(logger, "guardado matriz y vector de IDs"):
        np.save(config.F_MATRIZ_MACRO, dist_matrix)
        np.save(config.F_IDS_MACRO, ids_vuelos)
    logger.info(f"Matriz guardada en: {config.F_MATRIZ_MACRO}")
    logger.info(f"IDs guardados en:   {config.F_IDS_MACRO}")

    # Tamaño en disco (útil para discusión sobre escalabilidad)
    tam_mb = config.F_MATRIZ_MACRO.stat().st_size / (1024 * 1024)
    logger.info(f"Tamaño de la matriz en disco: {tam_mb:,.1f} MB")


if __name__ == "__main__":
    main()