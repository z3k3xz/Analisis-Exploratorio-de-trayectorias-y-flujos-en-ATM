from __future__ import annotations

import argparse

import hdbscan
import numpy as np
import pandas as pd

from src import config
from src.utils import (calcular_metricas_internas, configurar_logger,
                       cronometrar, guardar_metricas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-cluster-size', type=int, default=None,
                        help='Tamaño mínimo de cluster. Si no se pasa, se usa '
                             f'el {config.PCT_MIN_CLUSTER_SIZE_MACRO * 100:.1f}% '
                             'del total de vuelos.')
    parser.add_argument('--min-samples', type=int, default=config.MIN_SAMPLES_MACRO,
                        help='Parámetro min_samples de HDBSCAN.')
    args = parser.parse_args()

    logger = configurar_logger('06_clustering_macro')
    logger.info("=" * 60)
    logger.info(" CLUSTERING HDBSCAN — MACRO")
    logger.info("=" * 60)

    with cronometrar(logger, "carga matriz de distancias e IDs"):
        dist_matrix = np.load(config.F_MATRIZ_MACRO)
        ids_vuelos = np.load(config.F_IDS_MACRO, allow_pickle=True)
    n_vuelos = len(ids_vuelos)
    logger.info(f"Matriz: {dist_matrix.shape}, vuelos: {n_vuelos:,}")

    if args.min_cluster_size is None:
        mcs = max(int(n_vuelos * config.PCT_MIN_CLUSTER_SIZE_MACRO), 2)
        logger.info(f"min_cluster_size = {mcs} "
                    f"({config.PCT_MIN_CLUSTER_SIZE_MACRO * 100:.1f}% del total)")
    else:
        mcs = args.min_cluster_size
        logger.info(f"min_cluster_size = {mcs} (pasado por argumento)")
    logger.info(f"min_samples      = {args.min_samples}")

    with cronometrar(logger, "ejecución HDBSCAN"):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=args.min_samples,
            metric='precomputed',
        )
        etiquetas = clusterer.fit_predict(dist_matrix)

    with cronometrar(logger, "cálculo de métricas internas"):
        metricas = calcular_metricas_internas(dist_matrix, etiquetas)

    logger.info("")
    logger.info("Resultados del clustering:")
    logger.info(f"  Clusters identificados: {metricas['n_clusters']}")
    logger.info(f"  Vuelos en clusters:     {n_vuelos - metricas['n_ruido']:,} "
                f"({100 - metricas['pct_ruido']:.1f}%)")
    logger.info(f"  Vuelos como ruido:      {metricas['n_ruido']:,} "
                f"({metricas['pct_ruido']:.1f}%)")
    logger.info(f"  Cluster mayor:          {metricas['tamano_max_cluster']:,} vuelos "
                f"({metricas['pct_tamano_max_cluster']:.1f}%)")
    logger.info("")
    logger.info("Métricas internas:")
    logger.info(f"  Silhouette:         {metricas['silhouette']:>7.4f}   "
                f"(rango [-1, 1], más alto mejor)")
    logger.info(f"  Davies-Bouldin:     {metricas['davies_bouldin']:>7.4f}   "
                f"(rango [0, +∞), más bajo mejor)")
    logger.info(f"  Calinski-Harabasz:  {metricas['calinski_harabasz']:>7.1f}     "
                f"(rango [0, +∞), más alto mejor)")

    logger.info("")
    logger.info("Distribución por cluster:")
    df_resultado = pd.DataFrame({'flight_id': ids_vuelos, 'cluster': etiquetas})
    conteo = df_resultado['cluster'].value_counts().sort_index()
    for cid, count in conteo.items():
        etiqueta = "Ruido    " if cid == -1 else f"Cluster {cid:>2d}"
        pct = count / n_vuelos * 100
        logger.info(f"  {etiqueta}: {count:>5,} vuelos ({pct:>5.1f}%)")

    with cronometrar(logger, "guardado clusters y métricas"):
        df_resultado.to_parquet(config.F_CLUSTERS_MACRO, index=False)
        metricas_completas = {
            **metricas,
            'min_cluster_size': mcs,
            'min_samples': args.min_samples,
            'n_vuelos_entrada': n_vuelos,
        }
        guardar_metricas(metricas_completas, config.F_METRICAS_MACRO)
    logger.info(f"Clusters guardados en: {config.F_CLUSTERS_MACRO}")
    logger.info(f"Métricas guardadas en: {config.F_METRICAS_MACRO}")


if __name__ == "__main__":
    main()