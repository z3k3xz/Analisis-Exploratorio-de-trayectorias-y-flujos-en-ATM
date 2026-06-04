
from __future__ import annotations

import json

import hdbscan
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from src import config
from src.utils import (calcular_metricas_internas, configurar_logger,
                       cronometrar, preparar_matriz, remuestrear_trayectoria)


VALORES_N = [20, 30, 50, 80, 100]


def remuestrear_dataset(df: pd.DataFrame, n_puntos: int) -> pd.DataFrame:
    """Remuestrea todos los vuelos a n_puntos equidistantes."""
    resultados = []
    for fid, grupo in df.groupby('flight_id'):
        grupo = grupo.sort_values('timestamp')
        res = remuestrear_trayectoria(
            grupo['x'].values, grupo['y'].values, grupo['altitude'].values,
            n_puntos=n_puntos,
        )
        if res is None:
            continue
        x_i, y_i, alt_i = res
        resultados.append(pd.DataFrame({
            'flight_id': fid,
            'point_index': np.arange(n_puntos),
            'x': x_i,
            'y': y_i,
            'altitude': alt_i,
        }))
    if not resultados:
        return pd.DataFrame()
    return pd.concat(resultados, ignore_index=True)


def main():
    logger = configurar_logger('ablacion_n_puntos')
    logger.info("=" * 60)
    logger.info(" ABLACIÓN DE N PUNTOS DE REMUESTREO — MACRO")
    logger.info("=" * 60)

    with cronometrar(logger, "carga trayectorias proyectadas"):
        df = pd.read_parquet(config.F_PROYECTADO)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    n_vuelos_total = df['flight_id'].nunique()
    logger.info(f"Vuelos: {n_vuelos_total:,}")

    mcs = max(int(n_vuelos_total * config.PCT_MIN_CLUSTER_SIZE_MACRO), 2)
    logger.info(f"min_cluster_size fijo: {mcs} "
                f"({config.PCT_MIN_CLUSTER_SIZE_MACRO * 100:.1f}%)")
    logger.info(f"min_samples fijo: {config.MIN_SAMPLES_MACRO}")
    logger.info(f"Valores de N a probar: {VALORES_N}")
    logger.info("")

    resultados_globales = []

    for n_pts in VALORES_N:
        logger.info(f"{'=' * 40}")
        logger.info(f" N = {n_pts}")
        logger.info(f"{'=' * 40}")

        with cronometrar(logger, f"remuestreo a {n_pts} puntos"):
            df_rem = remuestrear_dataset(df, n_pts)
        n_vuelos = df_rem['flight_id'].nunique()
        logger.info(f"  Vuelos remuestreados: {n_vuelos:,}")

        with cronometrar(logger, f"matriz de distancias ({n_pts} pts)"):
            matriz, ids = preparar_matriz(df_rem)
            dist_condensada = pdist(matriz, metric='euclidean')
            dist_matrix = squareform(dist_condensada)
        logger.info(f"  Matriz: {matriz.shape}")

        with cronometrar(logger, f"HDBSCAN ({n_pts} pts)"):
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=config.MIN_SAMPLES_MACRO,
                metric='precomputed',
            )
            etiquetas = clusterer.fit_predict(dist_matrix)

        metricas = calcular_metricas_internas(dist_matrix, etiquetas)
        metricas['N'] = n_pts
        metricas['dimensiones'] = n_pts * 3
        metricas['n_vuelos'] = n_vuelos

        sil = metricas['silhouette']
        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
        logger.info(f"  Clusters: {metricas['n_clusters']}, "
                    f"Ruido: {metricas['pct_ruido']:.1f}%, "
                    f"Sil: {sil_str}")
        logger.info("")

        resultados_globales.append(metricas)

    # Tabla resumen
    logger.info("=" * 60)
    logger.info(" TABLA RESUMEN")
    logger.info("=" * 60)
    logger.info(f"{'N':>5} {'Dim':>5} {'Clusters':>8} {'Ruido%':>8} "
                f"{'Silhouette':>10} {'DB':>8} {'CH':>10}")
    logger.info("-" * 60)
    for r in resultados_globales:
        sil = r['silhouette']
        sil_s = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
        db = r['davies_bouldin']
        db_s = f"{db:.4f}" if not np.isnan(db) else "N/A"
        ch = r['calinski_harabasz']
        ch_s = f"{ch:.1f}" if not np.isnan(ch) else "N/A"
        logger.info(f"{r['N']:>5} {r['dimensiones']:>5} {r['n_clusters']:>8} "
                    f"{r['pct_ruido']:>7.1f}% {sil_s:>10} {db_s:>8} {ch_s:>10}")

    ruta = config.DIR_VALIDACIONES / "ablacion_n_puntos_macro.json"
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(resultados_globales, f, indent=2, ensure_ascii=False,
                  default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    logger.info(f"\nResultados guardados en: {ruta}")


if __name__ == "__main__":
    main()