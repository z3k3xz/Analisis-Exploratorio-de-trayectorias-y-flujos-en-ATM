
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from src import config
from src.utils import configurar_logger, coordenadas_aeropuerto_lcc
from src.validaciones.validacion_metering_fixes import METERING_FIXES


COLORES = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]
COLOR_RUIDO = '#d3d3d3'


def main():
    logger = configurar_logger('flujos_todos_loww')

    icao = "LOWW"
    esquema = "WED"

    df_tray = pd.read_parquet(config.f_micro_trayectorias(icao))
    df_clusters = pd.read_parquet(config.f_micro_clusters(icao, esquema))
    merged = df_tray.merge(df_clusters[['flight_id', 'cluster']],
                           on='flight_id', how='left')

    x_aero, y_aero = coordenadas_aeropuerto_lcc(icao)
    clusters_validos = sorted(merged[merged['cluster'] >= 0]['cluster'].unique())
    n_ruido = merged[merged['cluster'] == -1]['flight_id'].nunique()

    fig, ax = plt.subplots(figsize=(14, 12))

    # Ruido en gris
    if n_ruido > 0:
        sub = merged[merged['cluster'] == -1]
        for fid in sub['flight_id'].unique():
            v = sub[sub['flight_id'] == fid].sort_values('point_index')
            ax.plot(v['x'], v['y'], color=COLOR_RUIDO, alpha=0.15, linewidth=0.3)

    # Todas las trayectorias por cluster
    for c in clusters_validos:
        color = COLORES[c % len(COLORES)]
        sub = merged[merged['cluster'] == c]
        n = sub['flight_id'].nunique()
        first = True
        for fid in sub['flight_id'].unique():
            v = sub[sub['flight_id'] == fid].sort_values('point_index')
            if first:
                ax.plot(v['x'], v['y'], color=color, alpha=0.35,
                        linewidth=0.6, label=f'C{c} (n={n})')
                first = False
            else:
                ax.plot(v['x'], v['y'], color=color, alpha=0.35, linewidth=0.6)

    # Metering fixes
    transformer = Transformer.from_crs(config.CRS_ORIGEN, config.CRS_DESTINO,
                                       always_xy=True)
    for nombre, data in METERING_FIXES[icao].items():
        fx, fy = transformer.transform(data['lon'], data['lat'])
        ax.plot(fx, fy, marker='^', markersize=16, color='black',
                markerfacecolor='gold', markeredgewidth=1.5, zorder=10)
        ax.annotate(nombre, (fx, fy), textcoords="offset points",
                    xytext=(12, 10), fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.95))

    # Aeropuerto
    ax.plot(x_aero, y_aero, marker='*', markersize=24, color='red',
            markeredgecolor='black', markeredgewidth=1.5, zorder=11)
    ax.annotate(icao, (x_aero, y_aero), textcoords="offset points",
                xytext=(14, -18), fontsize=13, fontweight='bold', color='red')

    nombre_aer = config.COORDENADAS_AEROPUERTOS[icao]['nombre']
    n_total = merged['flight_id'].nunique()
    titulo = f"Flujos de llegada a {icao} ({nombre_aer})\n"
    titulo += f"Todas las trayectorias — esquema {esquema}, "
    titulo += f"{len(clusters_validos)} clusters, {n_ruido} vuelos como ruido"
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m, EPSG:3034 LCC)')
    ax.set_ylabel('Y (m, EPSG:3034 LCC)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left', ncol=2)

    plt.tight_layout()
    ruta_png = config.DIR_VALIDACIONES / "flujos_todos_LOWW_WED.png"
    plt.savefig(ruta_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figura guardada en: {ruta_png}")


if __name__ == "__main__":
    main()