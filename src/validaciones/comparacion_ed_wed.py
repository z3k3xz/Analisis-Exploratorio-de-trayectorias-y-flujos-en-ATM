
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.utils import configurar_logger, coordenadas_aeropuerto_lcc


COLORES = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]
COLOR_RUIDO = '#d3d3d3'


def dibujar_panel(ax, merged, x_aero, y_aero, icao, esquema):
    clusters_validos = sorted(merged[merged['cluster'] >= 0]['cluster'].unique())
    n_clusters = len(clusters_validos)
    n_ruido = merged[merged['cluster'] == -1]['flight_id'].nunique()
    n_total = merged['flight_id'].nunique()

    # Ruido
    if n_ruido > 0:
        sub = merged[merged['cluster'] == -1]
        for fid in sub['flight_id'].unique():
            v = sub[sub['flight_id'] == fid].sort_values('point_index')
            ax.plot(v['x'], v['y'], color=COLOR_RUIDO, alpha=0.15, linewidth=0.3)

    # Clusters
    for c in clusters_validos:
        color = COLORES[c % len(COLORES)]
        sub = merged[merged['cluster'] == c]
        n = sub['flight_id'].nunique()
        first = True
        for fid in sub['flight_id'].unique():
            v = sub[sub['flight_id'] == fid].sort_values('point_index')
            if first:
                ax.plot(v['x'], v['y'], color=color, alpha=0.35,
                        linewidth=0.6, label=f'C{c} ({n})')
                first = False
            else:
                ax.plot(v['x'], v['y'], color=color, alpha=0.35, linewidth=0.6)

    # Aeropuerto
    ax.plot(x_aero, y_aero, marker='*', markersize=18, color='red',
            markeredgecolor='black', markeredgewidth=1.2, zorder=11)

    ax.set_title(f'{esquema} — {n_clusters} clusters, {n_ruido} ruido '
                 f'({n_ruido/n_total*100:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', ncol=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('icao', type=str, default='LOWW', nargs='?')
    args = parser.parse_args()

    icao = args.icao.strip().upper()
    logger = configurar_logger(f'comparacion_ed_wed_{icao}')

    df_tray = pd.read_parquet(config.f_micro_trayectorias(icao))
    df_ed = pd.read_parquet(config.f_micro_clusters(icao, 'ED'))
    df_wed = pd.read_parquet(config.f_micro_clusters(icao, 'WED'))

    merged_ed = df_tray.merge(df_ed[['flight_id', 'cluster']],
                              on='flight_id', how='left')
    merged_wed = df_tray.merge(df_wed[['flight_id', 'cluster']],
                               on='flight_id', how='left')

    x_aero, y_aero = coordenadas_aeropuerto_lcc(icao)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    dibujar_panel(ax1, merged_ed, x_aero, y_aero, icao, 'ED')
    dibujar_panel(ax2, merged_wed, x_aero, y_aero, icao, 'WED')

    nombre = config.COORDENADAS_AEROPUERTOS[icao]['nombre']
    fig.suptitle(f'Comparación ED vs WED — {icao} ({nombre})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    ruta = config.DIR_VALIDACIONES / f"comparacion_ed_wed_{icao}.png"
    plt.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figura guardada en: {ruta}")


if __name__ == "__main__":
    main()