
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from src import config
from src.utils import (aplicar_pesos_a_matriz, configurar_logger,
                       cronometrar, preparar_matriz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('icao', type=str, help='Código ICAO del aeropuerto.')
    parser.add_argument('--esquema', type=str, default=config.ESQUEMA_DEFAULT,
                        choices=config.ESQUEMAS_DISTANCIA,
                        help='Esquema de distancia a aplicar.')
    args = parser.parse_args()

    icao = args.icao.strip().upper()
    esquema = args.esquema

    logger = configurar_logger(f'09_distancias_micro_{icao}_{esquema}')
    logger.info("=" * 60)
    logger.info(f" MATRIZ DE DISTANCIAS MICRO — {icao} — esquema: {esquema}")
    logger.info("=" * 60)

    ruta_tray = config.f_micro_trayectorias(icao)
    if not ruta_tray.exists():
        logger.error(f"No existe el fichero de trayectorias micro: {ruta_tray}")
        logger.error("Ejecuta primero: python -m src.micro.recorte_micro")
        return

    with cronometrar(logger, "carga trayectorias micro"):
        df = pd.read_parquet(ruta_tray)
    n_vuelos = df['flight_id'].nunique()
    logger.info(f"Vuelos cargados: {n_vuelos:,}")

    with cronometrar(logger, "construcción matriz vectorizada"):
        matriz, ids_vuelos = preparar_matriz(df)
    logger.info(f"Matriz: {matriz.shape}")

    with cronometrar(logger, f"aplicación de pesos ({esquema})"):
        matriz_ponderada = aplicar_pesos_a_matriz(matriz, esquema)

    n_pares = n_vuelos * (n_vuelos - 1) // 2
    logger.info(f"Pares a calcular: {n_pares:,}")

    with cronometrar(logger, "pdist(euclidean) + squareform"):
        dist_condensada = pdist(matriz_ponderada, metric='euclidean')
        dist_matrix = squareform(dist_condensada)

    tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    logger.info(f"Distancias ({esquema}):")
    logger.info(f"  Mín:     {tri.min():>15,.0f}")
    logger.info(f"  P05:     {np.percentile(tri, 5):>15,.0f}")
    logger.info(f"  Mediana: {np.median(tri):>15,.0f}")
    logger.info(f"  Media:   {tri.mean():>15,.0f}")
    logger.info(f"  P95:     {np.percentile(tri, 95):>15,.0f}")
    logger.info(f"  Máx:     {tri.max():>15,.0f}")

    ruta_matriz = config.f_micro_matriz(icao, esquema)
    ruta_ids = config.f_micro_ids(icao)
    np.save(ruta_matriz, dist_matrix)
    np.save(ruta_ids, ids_vuelos)
    logger.info(f"Matriz guardada en: {ruta_matriz}")
    logger.info(f"IDs guardados en:   {ruta_ids}")


if __name__ == "__main__":
    main()