from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from src import config
from src.utils import configurar_logger, cronometrar


def main():
    logger = configurar_logger('visualizar_macro')
    logger.info("=" * 60)
    logger.info(" VISUALIZACIÓN ESTÁTICA — MACRO")
    logger.info("=" * 60)

    with cronometrar(logger, "carga de datos"):
        df = pd.read_parquet(config.F_PROYECTADO)
        df_clusters = pd.read_parquet(config.F_CLUSTERS_MACRO)

    df = df.merge(df_clusters[['flight_id', 'cluster']], on='flight_id',
                  how='inner')

    df_ruido = df[df['cluster'] == -1]
    df_validos = df[df['cluster'] >= 0]

    n_clusters = df_validos['cluster'].nunique()
    n_ruido = df_ruido['flight_id'].nunique()
    n_validos = df_validos['flight_id'].nunique()

    logger.info(f"Vuelos en clusters: {n_validos:,}")
    logger.info(f"Vuelos ruido: {n_ruido:,}")
    logger.info(f"Clusters: {n_clusters}")

    colores = cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

    fig, ax = plt.subplots(figsize=(16, 12))

    # Ruido de fondo
    for fid in df_ruido['flight_id'].unique()[:500]:
        v = df_ruido[df_ruido['flight_id'] == fid].sort_values('timestamp')
        ax.plot(v['x'], v['y'], color='lightgray', alpha=0.1, linewidth=0.3)

    # Clusters
    for cid in sorted(df_validos['cluster'].unique()):
        sub = df_validos[df_validos['cluster'] == cid]
        color = colores[cid % len(colores)]
        n = sub['flight_id'].nunique()
        first = True
        for fid in sub['flight_id'].unique():
            v = sub[sub['flight_id'] == fid].sort_values('timestamp')
            if first:
                ax.plot(v['x'], v['y'], color=color, alpha=0.3,
                        linewidth=0.5, label=f'C{cid} ({n})')
                first = False
            else:
                ax.plot(v['x'], v['y'], color=color, alpha=0.3, linewidth=0.5)

    ax.set_title(f'Corredores macro — {n_clusters} clusters, '
                 f'{n_ruido} ruido', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m, EPSG:3034 LCC)')
    ax.set_ylabel('Y (m, EPSG:3034 LCC)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', ncol=2)

    plt.tight_layout()
    ruta = config.DIR_MACRO / "corredores_macro.png"
    plt.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figura guardada en: {ruta}")


if __name__ == "__main__":
    main()