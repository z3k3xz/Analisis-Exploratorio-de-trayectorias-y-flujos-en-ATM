
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config
from src.utils import (configurar_logger, coordenadas_aeropuerto_lcc,
                       cronometrar)


SECTORES = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def sector_cardinal(bearing_deg: float) -> str:
    """Convierte un rumbo en grados [0, 360) a uno de los 8 sectores."""
    idx = int(((bearing_deg + 22.5) % 360) / 45)
    return SECTORES[idx]


def caracterizar_cluster(df_tray: pd.DataFrame, df_meta: pd.DataFrame,
                         cluster_id: int, vuelos: np.ndarray,
                         x_aero: float, y_aero: float) -> dict:
    n = len(vuelos)
    r = {'cluster': int(cluster_id), 'n_vuelos': int(n)}

    meta = df_meta[df_meta['flight_id'].isin(vuelos)]

    # Orígenes
    if 'adep' in meta.columns:
        orig = meta['adep'].value_counts()
        r['top_origenes'] = ', '.join([f"{a} ({c})" for a, c in orig.head(5).items()])
        r['n_origenes_distintos'] = int(meta['adep'].nunique())
    else:
        r['top_origenes'] = 'N/A'
        r['n_origenes_distintos'] = 0

    # Países de origen
    if 'country_code_adep' in meta.columns:
        paises = meta['country_code_adep'].value_counts()
        r['top_paises_origen'] = ', '.join([f"{p} ({c})" for p, c in paises.head(5).items()])
        r['n_paises_distintos'] = int(meta['country_code_adep'].nunique())
    else:
        r['top_paises_origen'] = 'N/A'
        r['n_paises_distintos'] = 0

    # Aerolíneas
    if 'airline' in meta.columns:
        al = meta['airline'].value_counts()
        r['top_aerolineas'] = ', '.join([f"{a} ({c})" for a, c in al.head(5).items()])
    else:
        r['top_aerolineas'] = 'N/A'

    # Tipos de aeronave
    if 'aircraft_type' in meta.columns:
        ac = meta['aircraft_type'].value_counts()
        r['top_aeronaves'] = ', '.join([f"{a} ({c})" for a, c in ac.head(5).items()])
    else:
        r['top_aeronaves'] = 'N/A'

    # Sector de entrada al TMA: rumbo desde el aeropuerto al punto 0 del cluster
    primeros = df_tray[
        (df_tray['flight_id'].isin(vuelos)) &
        (df_tray['point_index'] == 0)
    ]
    if not primeros.empty:
        dx = primeros['x'].values - x_aero
        dy = primeros['y'].values - y_aero
        # En LCC: x crece al este, y crece al norte (igual que coordenadas geográficas).
        # Bearing convencional: 0° = N, 90° = E, sentido horario.
        bearing = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        sectores = pd.Series([sector_cardinal(b) for b in bearing])
        conteo_sec = sectores.value_counts()
        r['sector_principal'] = conteo_sec.index[0]
        r['pct_sector_principal'] = float(conteo_sec.iloc[0] / len(sectores) * 100)
        r['distribucion_sectores'] = ', '.join(
            [f"{s} ({c})" for s, c in conteo_sec.items()]
        )
        # Coordenadas medianas del punto 0 (entrada al TMA) en LCC
        r['entrada_x_mediana'] = float(np.median(primeros['x'].values))
        r['entrada_y_mediana'] = float(np.median(primeros['y'].values))
    else:
        r['sector_principal'] = '?'
        r['pct_sector_principal'] = 0.0
        r['distribucion_sectores'] = 'N/A'
        r['entrada_x_mediana'] = 0.0
        r['entrada_y_mediana'] = 0.0

    # Altitud al entrar y al aterrizar (altura ARP no se resta; valores absolutos en m)
    df_c = df_tray[df_tray['flight_id'].isin(vuelos)]
    alt_p0 = df_c[df_c['point_index'] == 0]['altitude']
    alt_pN = df_c[df_c['point_index'] == config.N_PUNTOS - 1]['altitude']
    r['alt_entrada_mediana_m'] = float(alt_p0.median()) if not alt_p0.empty else 0.0
    r['alt_final_mediana_m'] = float(alt_pN.median()) if not alt_pN.empty else 0.0

    return r


def procesar_aeropuerto(icao: str, esquema: str, logger) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info(f" CARACTERIZACIÓN MICRO — {icao} — esquema: {esquema}")
    logger.info("=" * 60)

    ruta_tray = config.f_micro_trayectorias(icao)
    ruta_clusters = config.f_micro_clusters(icao, esquema)
    if not ruta_tray.exists() or not ruta_clusters.exists():
        logger.error(f"Faltan ficheros para {icao}/{esquema}. Ejecuta antes el "
                     f"pipeline micro completo.")
        return

    df_tray = pd.read_parquet(ruta_tray)
    df_clusters = pd.read_parquet(ruta_clusters)
    df_meta = pd.read_csv(config.RUTA_METADATOS)

    x_aero, y_aero = coordenadas_aeropuerto_lcc(icao)
    logger.info(f"ARP LCC: ({x_aero:,.0f}, {y_aero:,.0f}) m")

    clusters_validos = sorted(df_clusters[df_clusters['cluster'] >= 0]['cluster'].unique())
    logger.info(f"Caracterizando {len(clusters_validos)} clusters + ruido")
    logger.info("")

    resumenes = []
    for cid in clusters_validos:
        vuelos = df_clusters[df_clusters['cluster'] == cid]['flight_id'].values
        r = caracterizar_cluster(df_tray, df_meta, cid, vuelos, x_aero, y_aero)
        r['icao'] = icao
        r['esquema'] = esquema
        resumenes.append(r)

        logger.info(f"--- Cluster {cid} ({r['n_vuelos']} vuelos) ---")
        logger.info(f"  Sector entrada TMA: {r['sector_principal']} "
                    f"({r['pct_sector_principal']:.0f}% de los vuelos)")
        logger.info(f"  Distribución sectores: {r['distribucion_sectores']}")
        logger.info(f"  Top orígenes:    {r['top_origenes']}")
        logger.info(f"  Top países:      {r['top_paises_origen']}")
        logger.info(f"  Top aerolíneas:  {r['top_aerolineas']}")
        logger.info(f"  Top aeronaves:   {r['top_aeronaves']}")
        logger.info(f"  Alt entrada / final (mediana, m): "
                    f"{r['alt_entrada_mediana_m']:.0f} / {r['alt_final_mediana_m']:.0f}")
        logger.info("")

    # Ruido
    ruido_ids = df_clusters[df_clusters['cluster'] == -1]['flight_id'].values
    if len(ruido_ids):
        r = caracterizar_cluster(df_tray, df_meta, -1, ruido_ids, x_aero, y_aero)
        r['icao'] = icao
        r['esquema'] = esquema
        resumenes.append(r)
        logger.info(f"--- RUIDO ({r['n_vuelos']} vuelos) ---")
        logger.info(f"  Sectores: {r['distribucion_sectores']}")
        logger.info(f"  Top orígenes: {r['top_origenes']}")
        logger.info("")

    # Parquet consolidado por aeropuerto + esquema
    df_resumen = pd.DataFrame(resumenes)
    ruta_salida = config.DIR_MICRO / f"caracterizacion_micro_{icao}_{esquema}.parquet"
    df_resumen.to_parquet(ruta_salida, index=False)
    logger.info(f"Resumen guardado en: {ruta_salida}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aeropuertos', nargs='+', default=config.AEROPUERTOS_MICRO)
    parser.add_argument('--esquema', type=str, default=config.ESQUEMA_DEFAULT,
                        choices=config.ESQUEMAS_DISTANCIA)
    args = parser.parse_args()

    logger = configurar_logger(f'11_caracterizacion_micro_{args.esquema}')
    logger.info("=" * 60)
    logger.info(" CARACTERIZACIÓN MICRO")
    logger.info("=" * 60)
    logger.info(f"Aeropuertos: {args.aeropuertos}")
    logger.info(f"Esquema:     {args.esquema}")

    for icao in args.aeropuertos:
        procesar_aeropuerto(icao, args.esquema, logger)


if __name__ == "__main__":
    main()