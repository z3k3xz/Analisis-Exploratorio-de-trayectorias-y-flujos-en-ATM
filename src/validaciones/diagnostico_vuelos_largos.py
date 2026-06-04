
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config
from src.utils import configurar_logger


UMBRAL_KM = 20_000   # km, > circunferencia terrestre / 2


def distancia_recorrida_lcc(grupo: pd.DataFrame) -> float:
    """Distancia 3D acumulada en km, sobre coordenadas LCC."""
    grupo = grupo.sort_values('timestamp')
    dx = np.diff(grupo['x'].values)
    dy = np.diff(grupo['y'].values)
    dalt = np.diff(grupo['altitude'].values)
    return float(np.sum(np.sqrt(dx**2 + dy**2 + dalt**2))) / 1000


def distancia_recorrida_wgs(grupo: pd.DataFrame) -> float:
    """Distancia recorrida en WGS84, fórmula haversine acumulada (km).
    Útil como referencia: si LCC da 60.000 km pero haversine da 8.000 km,
    el problema está en la proyección, no en los datos."""
    grupo = grupo.sort_values('timestamp')
    lat = np.radians(grupo['latitude'].values)
    lon = np.radians(grupo['longitude'].values)
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return float(np.sum(6371.0 * c))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--umbral-km', type=float, default=UMBRAL_KM,
                        help=f'Umbral en km para considerar un vuelo sospechoso '
                             f'(default {UMBRAL_KM}).')
    args = parser.parse_args()

    logger = configurar_logger('diagnostico_vuelos_largos')
    logger.info("=" * 60)
    logger.info(" DIAGNÓSTICO DE VUELOS CON DISTANCIA EXCESIVA")
    logger.info("=" * 60)

    # Cargar datos antes y después de proyectar
    logger.info("Cargando datos limpios (WGS84) y proyectados (LCC)...")
    df_lim = pd.read_parquet(config.F_LIMPIO,
                             columns=['flight_id', 'timestamp',
                                      'latitude', 'longitude', 'altitude'])
    df_lim['timestamp'] = pd.to_datetime(df_lim['timestamp'], utc=True)
    df_proy = pd.read_parquet(config.F_PROYECTADO)
    df_proy['timestamp'] = pd.to_datetime(df_proy['timestamp'], utc=True)
    df_meta = pd.read_csv(config.RUTA_METADATOS)

    logger.info(f"Vuelos en datos proyectados: {df_proy['flight_id'].nunique():,}")

    # Calcular distancia LCC por vuelo
    logger.info(f"Calculando distancia recorrida LCC por vuelo...")
    dist_lcc = (df_proy.groupby('flight_id')
                .apply(distancia_recorrida_lcc, include_groups=False)
                .reset_index(name='dist_lcc_km'))

    sospechosos = dist_lcc[dist_lcc['dist_lcc_km'] > args.umbral_km]
    logger.info(f"Vuelos con dist LCC > {args.umbral_km:,} km: {len(sospechosos)}")

    if len(sospechosos) == 0:
        logger.info("Ningún vuelo sospechoso. Nada que diagnosticar.")
        return

    logger.info("")
    logger.info("Detalle por vuelo sospechoso:")
    logger.info("-" * 60)

    for _, fila in sospechosos.sort_values('dist_lcc_km', ascending=False).iterrows():
        fid = fila['flight_id']
        d_lcc = fila['dist_lcc_km']

        # Distancia haversine sobre WGS84 (referencia "real")
        grupo_lim = df_lim[df_lim['flight_id'] == fid]
        d_wgs = distancia_recorrida_wgs(grupo_lim)

        # Bounding box en WGS84
        lat_min, lat_max = grupo_lim['latitude'].min(), grupo_lim['latitude'].max()
        lon_min, lon_max = grupo_lim['longitude'].min(), grupo_lim['longitude'].max()

        # Metadatos
        meta = df_meta[df_meta['flight_id'] == fid]
        if len(meta):
            adep = meta['adep'].iloc[0]
            ades = meta['ades'].iloc[0]
            ac = meta.get('aircraft_type', pd.Series(['?'])).iloc[0] if 'aircraft_type' in meta.columns else '?'
        else:
            adep = ades = ac = '?'

        # Diagnóstico tentativo
        cruza_antimeridiano = (lon_max - lon_min) > 180
        fuera_europa = (lat_min < 20) or (lat_max > 72) or (lon_min < -25) or (lon_max > 50)
        ratio_lcc_wgs = d_lcc / d_wgs if d_wgs > 0 else float('inf')

        logger.info(f"flight_id = {fid}")
        logger.info(f"  Ruta:        {adep} → {ades}  ({ac})")
        logger.info(f"  Bbox WGS84:  lat [{lat_min:.2f}, {lat_max:.2f}], "
                    f"lon [{lon_min:.2f}, {lon_max:.2f}]")
        logger.info(f"  Distancia:   LCC = {d_lcc:>10,.1f} km   |   "
                    f"Haversine WGS84 = {d_wgs:>8,.1f} km   |   "
                    f"ratio LCC/WGS = {ratio_lcc_wgs:.1f}x")
        logger.info(f"  Cruza antimeridiano (Δlon > 180°): {cruza_antimeridiano}")
        logger.info(f"  Fuera bbox Europa (lat [20,72], lon [-25,50]): {fuera_europa}")
        logger.info(f"  Puntos: {len(grupo_lim):,}")
        logger.info("")


if __name__ == "__main__":
    main()