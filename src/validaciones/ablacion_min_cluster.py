
from __future__ import annotations

import argparse
import json

import hdbscan
import numpy as np
import pandas as pd

from src import config
from src.utils import calcular_metricas_internas, configurar_logger, cronometrar


def ejecutar_barrido(dist_matrix: np.ndarray, valores_mcs: list[int],
                     min_samples: int, logger) -> list[dict]:
    resultados = []
    for mcs in valores_mcs:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=min_samples,
            metric='precomputed',
        )
        etiquetas = clusterer.fit_predict(dist_matrix)
        metricas = calcular_metricas_internas(dist_matrix, etiquetas)
        metricas['min_cluster_size'] = mcs
        resultados.append(metricas)

        sil = metricas['silhouette']
        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
        logger.info(f"  mcs={mcs:>4d}  →  clusters={metricas['n_clusters']:>3d}  "
                    f"ruido={metricas['pct_ruido']:>5.1f}%  "
                    f"Sil={sil_str}  "
                    f"DB={metricas['davies_bouldin']:.4f}  "
                    f"CH={metricas['calinski_harabasz']:.1f}")
    return resultados


def ablacion_macro(args, logger):
    logger.info("=" * 60)
    logger.info(" ABLACIÓN min_cluster_size — MACRO")
    logger.info("=" * 60)

    dist_matrix = np.load(config.F_MATRIZ_MACRO)
    n = dist_matrix.shape[0]
    logger.info(f"Vuelos: {n:,}")

    # Rango: de 0.5% a 5% del total, en pasos razonables
    pcts = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    valores = sorted(set(max(int(n * p), 2) for p in pcts))
    logger.info(f"Valores de min_cluster_size a probar: {valores}")
    logger.info(f"min_samples fijo: {config.MIN_SAMPLES_MACRO}")
    logger.info("")

    with cronometrar(logger, "barrido completo"):
        resultados = ejecutar_barrido(
            dist_matrix, valores, config.MIN_SAMPLES_MACRO, logger
        )

    ruta = config.DIR_VALIDACIONES / "ablacion_min_cluster_macro.json"
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False,
                  default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    logger.info(f"\nResultados guardados en: {ruta}")


def ablacion_micro(args, logger):
    icao = args.icao.strip().upper()
    esquema = args.esquema

    logger.info("=" * 60)
    logger.info(f" ABLACIÓN min_cluster_size — MICRO {icao} — esquema: {esquema}")
    logger.info("=" * 60)

    ruta_matriz = config.f_micro_matriz(icao, esquema)
    if not ruta_matriz.exists():
        logger.error(f"No existe: {ruta_matriz}")
        return

    dist_matrix = np.load(ruta_matriz)
    n = dist_matrix.shape[0]
    logger.info(f"Vuelos: {n:,}")

    # Rango: de 0.5% a 10% (micro tiene menos vuelos, necesita rango más amplio)
    pcts = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10]
    valores = sorted(set(max(int(n * p), 2) for p in pcts))
    logger.info(f"Valores de min_cluster_size a probar: {valores}")
    logger.info(f"min_samples fijo: {config.MIN_SAMPLES_MICRO}")
    logger.info("")

    with cronometrar(logger, "barrido completo"):
        resultados = ejecutar_barrido(
            dist_matrix, valores, config.MIN_SAMPLES_MICRO, logger
        )

    ruta = config.DIR_VALIDACIONES / f"ablacion_min_cluster_micro_{icao}_{esquema}.json"
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False,
                  default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    logger.info(f"\nResultados guardados en: {ruta}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nivel', required=True, choices=['macro', 'micro'])
    parser.add_argument('--icao', type=str, default='LOWW',
                        help='Solo para micro. ICAO del aeropuerto.')
    parser.add_argument('--esquema', type=str, default=config.ESQUEMA_DEFAULT,
                        choices=config.ESQUEMAS_DISTANCIA)
    args = parser.parse_args()

    nombre_log = f"ablacion_mcs_{args.nivel}"
    if args.nivel == 'micro':
        nombre_log += f"_{args.icao.upper()}_{args.esquema}"
    logger = configurar_logger(nombre_log)

    if args.nivel == 'macro':
        ablacion_macro(args, logger)
    else:
        ablacion_micro(args, logger)


if __name__ == "__main__":
    main()