
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config
from src.utils import (configurar_logger, coordenadas_aeropuerto_lcc,
                       cronometrar, remuestrear_trayectoria)


def recortar_y_remuestrear(df_tray: pd.DataFrame, ids_destino: list,
                           x_aero: float, y_aero: float,
                           radio: float, n_puntos: int,
                           logger) -> tuple[pd.DataFrame, int, int]:

    df = df_tray[df_tray['flight_id'].isin(ids_destino)].copy()
    df['dist_aero'] = np.sqrt((df['x'] - x_aero) ** 2 + (df['y'] - y_aero) ** 2)
    df = df[df['dist_aero'] <= radio]

    n_descartados_radio = 0
    n_descartados_remuestreo = 0
    resultados = []

    for fid, grupo in df.groupby('flight_id'):
        if len(grupo) < 10:
            n_descartados_radio += 1
            continue

        grupo = grupo.sort_values('timestamp')
        res = remuestrear_trayectoria(
            grupo['x'].values,
            grupo['y'].values,
            grupo['altitude'].values,
            n_puntos=n_puntos,
        )
        if res is None:
            n_descartados_remuestreo += 1
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
        return pd.DataFrame(), n_descartados_radio, n_descartados_remuestreo

    return pd.concat(resultados, ignore_index=True), n_descartados_radio, n_descartados_remuestreo


def procesar_aeropuerto(icao: str, df_tray: pd.DataFrame,
                        df_meta: pd.DataFrame, logger) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info(f" AEROPUERTO: {icao} ({config.COORDENADAS_AEROPUERTOS[icao]['nombre']})")
    logger.info("=" * 60)

    # Posición oficial proyectada
    x_aero, y_aero = coordenadas_aeropuerto_lcc(icao)
    logger.info(f"ARP oficial (WGS84): "
                f"lat {config.COORDENADAS_AEROPUERTOS[icao]['lat']:.5f}, "
                f"lon {config.COORDENADAS_AEROPUERTOS[icao]['lon']:.5f}")
    logger.info(f"ARP proyectado (LCC): ({x_aero:,.0f}, {y_aero:,.0f}) m")

    # Vuelos con destino en este aeropuerto
    ids_destino = df_meta[df_meta['ades'] == icao]['flight_id'].values
    ids_presentes = df_tray['flight_id'].unique()
    ids_destino = [fid for fid in ids_destino if fid in ids_presentes]
    logger.info(f"Vuelos con destino {icao} en el dataset limpio: {len(ids_destino):,}")

    if len(ids_destino) == 0:
        logger.warning(f"No hay vuelos con destino {icao}. Se omite.")
        return

    with cronometrar(logger, f"recorte + remuestreo {icao}"):
        df_micro, n_desc_radio, n_desc_remuestreo = recortar_y_remuestrear(
            df_tray, ids_destino, x_aero, y_aero,
            config.RADIO_TERMINAL, config.N_PUNTOS, logger,
        )

    if df_micro.empty:
        logger.warning(f"No quedan vuelos micro para {icao}. Se omite.")
        return

    n_final = df_micro['flight_id'].nunique()
    logger.info(f"Vuelos con tramo terminal: {n_final:,}")
    logger.info(f"  Descartados por < 10 puntos en el TMA: {n_desc_radio}")
    logger.info(f"  Descartados en remuestreo:             {n_desc_remuestreo}")

    ruta_salida = config.f_micro_trayectorias(icao)
    df_micro.to_parquet(ruta_salida, index=False)
    logger.info(f"Guardado: {ruta_salida}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--aeropuertos', nargs='+', default=config.AEROPUERTOS_MICRO,
        help='ICAOs a procesar. Default: los declarados en config.AEROPUERTOS_MICRO.'
    )
    args = parser.parse_args()

    logger = configurar_logger('08_recorte_micro')
    logger.info("=" * 60)
    logger.info(" RECORTE Y REMUESTREO AL ÁREA TERMINAL")
    logger.info("=" * 60)
    logger.info(f"Aeropuertos a procesar: {args.aeropuertos}")
    logger.info(f"Radio terminal: {config.RADIO_TERMINAL / 1000:.0f} km")
    logger.info(f"Puntos por trayectoria: {config.N_PUNTOS}")

    with cronometrar(logger, "carga trayectorias proyectadas + metadatos"):
        df_tray = pd.read_parquet(config.F_PROYECTADO)
        df_tray['timestamp'] = pd.to_datetime(df_tray['timestamp'], utc=True)
        df_meta = pd.read_csv(config.RUTA_METADATOS)

    # Top destinos para contexto
    ids_presentes = df_tray['flight_id'].unique()
    meta_presentes = df_meta[df_meta['flight_id'].isin(ids_presentes)]
    top_destinos = meta_presentes['ades'].value_counts().head(20)
    logger.info("Top 20 destinos del dataset limpio:")
    for i, (icao, count) in enumerate(top_destinos.items(), 1):
        nombre = meta_presentes[meta_presentes['ades'] == icao]['name_ades'].iloc[0]
        flag = "  ← micro" if icao in args.aeropuertos else ""
        logger.info(f"  {i:>2d}. {icao} ({nombre}): {count:,} vuelos{flag}")

    for icao in args.aeropuertos:
        procesar_aeropuerto(icao, df_tray, df_meta, logger)


if __name__ == "__main__":
    main()