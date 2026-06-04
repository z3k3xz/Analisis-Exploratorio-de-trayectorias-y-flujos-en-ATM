
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.utils import configurar_logger, cronometrar


def main():
    logger = configurar_logger('analisis_ruido_macro')
    logger.info("=" * 60)
    logger.info(" SUB-ANÁLISIS DEL RUIDO MACRO")
    logger.info("=" * 60)

    with cronometrar(logger, "carga de datos"):
        df_clusters = pd.read_parquet(config.F_CLUSTERS_MACRO)
        df_meta = pd.read_csv(config.RUTA_METADATOS)

    vuelos_ruido = df_clusters[df_clusters['cluster'] == -1]['flight_id'].values
    vuelos_cluster = df_clusters[df_clusters['cluster'] >= 0]['flight_id'].values
    n_ruido = len(vuelos_ruido)
    n_cluster = len(vuelos_cluster)
    n_total = len(df_clusters)

    logger.info(f"Total: {n_total:,} vuelos")
    logger.info(f"En clusters: {n_cluster:,} ({n_cluster / n_total * 100:.1f}%)")
    logger.info(f"Ruido: {n_ruido:,} ({n_ruido / n_total * 100:.1f}%)")

    meta_ruido = df_meta[df_meta['flight_id'].isin(vuelos_ruido)]
    meta_cluster = df_meta[df_meta['flight_id'].isin(vuelos_cluster)]

    # --- Aeropuertos de origen ---
    logger.info("")
    logger.info("AEROPUERTOS DE ORIGEN — RUIDO vs CLUSTERS")
    logger.info("-" * 50)

    origenes_ruido = meta_ruido['adep'].value_counts().head(15)
    origenes_cluster = meta_cluster['adep'].value_counts().head(15)

    logger.info("Top 15 orígenes en RUIDO:")
    for i, (code, count) in enumerate(origenes_ruido.items(), 1):
        nombre = meta_ruido[meta_ruido['adep'] == code]['name_adep'].iloc[0]
        pct = count / n_ruido * 100
        logger.info(f"  {i:>2d}. {code} ({nombre}): {count} ({pct:.1f}%)")

    logger.info("")
    logger.info("Top 15 orígenes en CLUSTERS:")
    for i, (code, count) in enumerate(origenes_cluster.items(), 1):
        nombre = meta_cluster[meta_cluster['adep'] == code]['name_adep'].iloc[0]
        pct = count / n_cluster * 100
        logger.info(f"  {i:>2d}. {code} ({nombre}): {count} ({pct:.1f}%)")

    # --- Rutas ---
    logger.info("")
    logger.info("ANÁLISIS DE RUTAS — RUIDO vs CLUSTERS")
    logger.info("-" * 50)

    rutas_ruido = (meta_ruido['adep'].fillna('?') + ' → ' + meta_ruido['ades'].fillna('?')).value_counts()
    rutas_cluster = (meta_cluster['adep'].fillna('?') + ' → ' + meta_cluster['ades'].fillna('?')).value_counts()

    logger.info(f"Rutas únicas en ruido:    {len(rutas_ruido):,}")
    logger.info(f"Rutas únicas en clusters: {len(rutas_cluster):,}")
    logger.info(f"Rutas con 1 vuelo (ruido):    {(rutas_ruido == 1).sum()} "
                f"({(rutas_ruido == 1).sum() / len(rutas_ruido) * 100:.1f}%)")
    logger.info(f"Rutas con 1 vuelo (clusters): {(rutas_cluster == 1).sum()} "
                f"({(rutas_cluster == 1).sum() / len(rutas_cluster) * 100:.1f}%)")
    logger.info(f"Vuelos/ruta mediana (ruido):    {rutas_ruido.median():.0f}")
    logger.info(f"Vuelos/ruta mediana (clusters): {rutas_cluster.median():.0f}")

    logger.info("")
    logger.info("Top 10 rutas en RUIDO:")
    for i, (ruta, count) in enumerate(rutas_ruido.head(10).items(), 1):
        logger.info(f"  {i:>2d}. {ruta}: {count}")

    logger.info("")
    logger.info("Top 10 rutas en CLUSTERS:")
    for i, (ruta, count) in enumerate(rutas_cluster.head(10).items(), 1):
        logger.info(f"  {i:>2d}. {ruta}: {count}")

    # --- Dispersión geográfica ---
    logger.info("")
    logger.info("DISPERSIÓN GEOGRÁFICA")
    logger.info("-" * 50)

    n_orig_ruido = meta_ruido['adep'].nunique()
    n_dest_ruido = meta_ruido['ades'].nunique()
    n_orig_cluster = meta_cluster['adep'].nunique()
    n_dest_cluster = meta_cluster['ades'].nunique()

    logger.info(f"Ruido:    {n_orig_ruido} orígenes, {n_dest_ruido} destinos distintos")
    logger.info(f"Clusters: {n_orig_cluster} orígenes, {n_dest_cluster} destinos distintos")

    # --- Gráficas ---
    with cronometrar(logger, "generación de gráficas"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Top 10 orígenes ruido
        top_r = origenes_ruido.head(10)
        axes[0, 0].barh(range(len(top_r)), top_r.values, color='#d3d3d3')
        axes[0, 0].set_yticks(range(len(top_r)))
        axes[0, 0].set_yticklabels(top_r.index)
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel('Vuelos')
        axes[0, 0].set_title('Top 10 orígenes — RUIDO')

        # Top 10 orígenes clusters
        top_c = origenes_cluster.head(10)
        axes[0, 1].barh(range(len(top_c)), top_c.values, color='#1f77b4')
        axes[0, 1].set_yticks(range(len(top_c)))
        axes[0, 1].set_yticklabels(top_c.index)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel('Vuelos')
        axes[0, 1].set_title('Top 10 orígenes — CLUSTERS')

        # Distribución vuelos por ruta (ruido)
        axes[1, 0].hist(rutas_ruido.values, bins=30, color='#d3d3d3',
                        edgecolor='gray', alpha=0.8)
        axes[1, 0].set_xlabel('Vuelos por ruta')
        axes[1, 0].set_ylabel('Número de rutas')
        axes[1, 0].set_title(f'Distribución vuelos/ruta — RUIDO\n'
                             f'(mediana: {rutas_ruido.median():.0f})')
        axes[1, 0].axvline(rutas_ruido.median(), color='red', linestyle='--',
                           label=f'Mediana: {rutas_ruido.median():.0f}')
        axes[1, 0].legend()

        # Distribución vuelos por ruta (clusters)
        axes[1, 1].hist(rutas_cluster.values, bins=30, color='#1f77b4',
                        edgecolor='gray', alpha=0.8)
        axes[1, 1].set_xlabel('Vuelos por ruta')
        axes[1, 1].set_ylabel('Número de rutas')
        axes[1, 1].set_title(f'Distribución vuelos/ruta — CLUSTERS\n'
                             f'(mediana: {rutas_cluster.median():.0f})')
        axes[1, 1].axvline(rutas_cluster.median(), color='red', linestyle='--',
                           label=f'Mediana: {rutas_cluster.median():.0f}')
        axes[1, 1].legend()

        fig.suptitle('Sub-análisis del ruido macro: ¿qué contienen los vuelos '
                     'etiquetados como ruido?', fontsize=14, fontweight='bold')
        plt.tight_layout()

        ruta_png = config.DIR_VALIDACIONES / "analisis_ruido_macro.png"
        plt.savefig(ruta_png, dpi=180, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Figura guardada en: {ruta_png}")


if __name__ == "__main__":
    main()