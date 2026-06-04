
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import configurar_logger, cronometrar


def aplicar_filtros(df: pd.DataFrame, logger) -> pd.DataFrame:
    n_v_inicial = df['flight_id'].nunique()
    n_p_inicial = len(df)

    # Ordenación crítica antes de calcular diferencias por vuelo
    df = df.sort_values(['flight_id', 'timestamp']).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Filtro A: saltos de posición imposibles
    # ------------------------------------------------------------------
    df['dt'] = df.groupby('flight_id')['timestamp'].diff().dt.total_seconds()
    dlat = df.groupby('flight_id')['latitude'].diff()
    dlon = df.groupby('flight_id')['longitude'].diff()
    cos_lat = np.cos(np.radians(df['latitude']))
    dist_nm = np.sqrt((dlat * 60) ** 2 + (dlon * 60 * cos_lat) ** 2)
    dt_horas = df['dt'] / 3600
    vel = dist_nm / dt_horas
    mask = vel.fillna(0) <= config.VELOCIDAD_MAX_KT
    n_antes = len(df)
    df = df[mask]
    logger.info(f"  [A] Puntos con salto de posición imposible (>{config.VELOCIDAD_MAX_KT} kt): "
                f"{n_antes - len(df):,}")

    # ------------------------------------------------------------------
    # Filtro B: errores de altitud
    # Tasa de cambio vertical implausible (>100 ft/s ≈ 6000 ft/min).
    # ------------------------------------------------------------------
    df['dt'] = df.groupby('flight_id')['timestamp'].diff().dt.total_seconds()
    dalt = df.groupby('flight_id')['altitude'].diff().abs()
    cambio_alt_seg = dalt / df['dt']
    mask = cambio_alt_seg.fillna(0) <= config.CAMBIO_ALT_MAX
    n_antes = len(df)
    df = df[mask]
    logger.info(f"  [B] Puntos con cambio de altitud anómalo (>{config.CAMBIO_ALT_MAX} ft/s): "
                f"{n_antes - len(df):,}")

    # ------------------------------------------------------------------
    # Filtro C: rango operativo de altitud
    # ------------------------------------------------------------------
    n_antes = len(df)
    df = df[(df['altitude'] >= config.ALTITUD_MIN) &
            (df['altitude'] <= config.ALTITUD_MAX)]
    logger.info(f"  [C] Puntos fuera de rango [{config.ALTITUD_MIN}, "
                f"{config.ALTITUD_MAX}] ft: {n_antes - len(df):,}")

    # ------------------------------------------------------------------
    # Filtro D: vuelos con huecos temporales largos
    # ------------------------------------------------------------------
    df['dt'] = df.groupby('flight_id')['timestamp'].diff().dt.total_seconds()
    huecos_max = df.groupby('flight_id')['dt'].max()
    ids_validos = huecos_max[huecos_max <= config.UMBRAL_HUECO_MAX].index
    n_v_antes = df['flight_id'].nunique()
    df = df[df['flight_id'].isin(ids_validos)]
    logger.info(f"  [D] Vuelos descartados por hueco > "
                f"{config.UMBRAL_HUECO_MAX}s: "
                f"{n_v_antes - df['flight_id'].nunique():,}")

    # ------------------------------------------------------------------
    # Filtro E: mínimo de puntos por vuelo
    # ------------------------------------------------------------------
    conteo = df.groupby('flight_id').size()
    ids_suf = conteo[conteo >= config.MIN_PUNTOS].index
    n_v_antes = df['flight_id'].nunique()
    df = df[df['flight_id'].isin(ids_suf)]
    logger.info(f"  [E] Vuelos descartados por < {config.MIN_PUNTOS} puntos: "
                f"{n_v_antes - df['flight_id'].nunique():,}")

    # ------------------------------------------------------------------
    # Filtro F: dispersión geográfica anómala
    # ------------------------------------------------------------------
    bbox = df.groupby('flight_id').agg(
        lat_min=('latitude', 'min'),
        lat_max=('latitude', 'max'),
        lon_min=('longitude', 'min'),
        lon_max=('longitude', 'max'),
    )
    dentro = (
        (bbox['lat_min'] >= config.BBOX_LAT_MIN) &
        (bbox['lat_max'] <= config.BBOX_LAT_MAX) &
        (bbox['lon_min'] >= config.BBOX_LON_MIN) &
        (bbox['lon_max'] <= config.BBOX_LON_MAX)
    )
    ids_validos = bbox[dentro].index
    n_v_antes = df['flight_id'].nunique()
    df = df[df['flight_id'].isin(ids_validos)]
    logger.info(f"  [F] Vuelos descartados por bbox fuera de "
                f"[lat {config.BBOX_LAT_MIN}–{config.BBOX_LAT_MAX}, "
                f"lon {config.BBOX_LON_MIN}–{config.BBOX_LON_MAX}]: "
                f"{n_v_antes - df['flight_id'].nunique():,}")

    df = df.drop(columns=['dt']).reset_index(drop=True)

    n_v_final = df['flight_id'].nunique()
    n_p_final = len(df)
    logger.info("")
    logger.info(f"Resumen de limpieza:")
    logger.info(f"  Vuelos: {n_v_inicial:,} → {n_v_final:,} "
                f"(eliminados {n_v_inicial - n_v_final:,}, "
                f"{(n_v_inicial - n_v_final) / max(n_v_inicial, 1) * 100:.1f}%)")
    logger.info(f"  Puntos: {n_p_inicial:,} → {n_p_final:,} "
                f"(reducción {n_p_inicial / max(n_p_final, 1):.2f}x)")
    return df


def main():
    logger = configurar_logger('02_limpieza')
    logger.info("=" * 60)
    logger.info(" LIMPIEZA DE TRAYECTORIAS ADS-B")
    logger.info("=" * 60)

    with cronometrar(logger, "carga datos fusionados"):
        df = pd.read_parquet(config.F_FUSIONADO)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    logger.info(f"Cargados {len(df):,} puntos de {df['flight_id'].nunique():,} vuelos")

    with cronometrar(logger, "aplicación de filtros"):
        df_limpio = aplicar_filtros(df, logger)

    with cronometrar(logger, "escritura trayectorias limpias"):
        df_limpio.to_parquet(config.F_LIMPIO, index=False)
    logger.info(f"Guardado en: {config.F_LIMPIO}")


if __name__ == "__main__":
    main()