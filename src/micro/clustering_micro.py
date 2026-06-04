
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
    parser.add_argument('icao', type=str, help='Código ICAO del aeropuerto.')
    parser.add_argument('--esquema', type=str, default=config.ESQUEMA_DEFAULT,
                        choices=config.ESQUEMAS_DISTANCIA)
    parser.add_argument('--min-cluster-size', type=int, default=None,
                        help='Si no se pasa, se usa el '
                             f'{config.PCT_MIN_CLUSTER_SIZE_MICRO * 100:.1f}% del total.')
    parser.add_argument('--min-samples', type=int, default=config.MIN_SAMPLES_MICRO)
    args = parser.parse_args()

    icao = args.icao.strip().upper()
    esquema = args.esquema

    logger = configurar_logger(f'10_clustering_micro_{icao}_{esquema}')
    logger.info("=" * 60)
    logger.info(f" CLUSTERING HDBSCAN MICRO — {icao} — esquema: {esquema}")
    logger.info("=" * 60)

    ruta_matriz = config.f_micro_matriz(icao, esquema)
    ruta_ids = config.f_micro_ids(icao)
    if not ruta_matriz.exists():
        logger.error(f"No existe la matriz: {ruta_matriz}")
        logger.error(f"Ejecuta primero: "
                     f"python -m src.micro.distancias_micro {icao} --esquema {esquema}")
        return

    with cronometrar(logger, "carga matriz e IDs"):
        dist_matrix = np.load(ruta_matriz)
        ids_vuelos = np.load(ruta_ids, allow_pickle=True)
    n_vuelos = len(ids_vuelos)
    logger.info(f"Matriz: {dist_matrix.shape}, vuelos: {n_vuelos:,}")

    if args.min_cluster_size is None:
        mcs = max(int(n_vuelos * config.PCT_MIN_CLUSTER_SIZE_MICRO), 2)
        logger.info(f"min_cluster_size = {mcs} "
                    f"({config.PCT_MIN_CLUSTER_SIZE_MICRO * 100:.1f}% del total)")
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
    logger.info("Resultados:")
    logger.info(f"  Clusters identificados: {metricas['n_clusters']}")
    logger.info(f"  Vuelos en clusters:     {n_vuelos - metricas['n_ruido']:,} "
                f"({100 - metricas['pct_ruido']:.1f}%)")
    logger.info(f"  Vuelos como ruido:      {metricas['n_ruido']:,} "
                f"({metricas['pct_ruido']:.1f}%)")
    logger.info(f"  Cluster mayor:          {metricas['tamano_max_cluster']:,} "
                f"({metricas['pct_tamano_max_cluster']:.1f}%)")
    logger.info("")
    logger.info("Métricas internas:")
    logger.info(f"  Silhouette:         {metricas['silhouette']:>7.4f}")
    logger.info(f"  Davies-Bouldin:     {metricas['davies_bouldin']:>7.4f}")
    logger.info(f"  Calinski-Harabasz:  {metricas['calinski_harabasz']:>7.1f}")

    logger.info("")
    logger.info("Distribución por cluster:")
    df_resultado = pd.DataFrame({'flight_id': ids_vuelos, 'cluster': etiquetas})
    conteo = df_resultado['cluster'].value_counts().sort_index()
    for cid, count in conteo.items():
        etiqueta = "Ruido    " if cid == -1 else f"Cluster {cid:>2d}"
        pct = count / n_vuelos * 100
        logger.info(f"  {etiqueta}: {count:>4,} ({pct:>5.1f}%)")

    ruta_clusters = config.f_micro_clusters(icao, esquema)
    ruta_metricas = config.f_micro_metricas(icao, esquema)
    df_resultado.to_parquet(ruta_clusters, index=False)
    guardar_metricas({
        **metricas,
        'icao': icao,
        'esquema': esquema,
        'min_cluster_size': mcs,
        'min_samples': args.min_samples,
        'n_vuelos_entrada': n_vuelos,
    }, ruta_metricas)
    logger.info(f"Clusters guardados en: {ruta_clusters}")
    logger.info(f"Métricas guardadas en: {ruta_metricas}")


if __name__ == "__main__":
    main()