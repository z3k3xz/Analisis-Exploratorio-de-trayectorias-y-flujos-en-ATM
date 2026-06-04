
from __future__ import annotations

import pandas as pd
from pyproj import Transformer

from src import config
from src.utils import configurar_logger, cronometrar


def proyectar(df: pd.DataFrame, transformer: Transformer) -> pd.DataFrame:
    # always_xy=True: entrada (lon, lat), salida (x, y) en convención cartesiana.
    x, y = transformer.transform(df['longitude'].values, df['latitude'].values)
    alt_m = df['altitude'].values * config.PIES_A_METROS
    return pd.DataFrame({
        'flight_id': df['flight_id'].values,
        'timestamp': df['timestamp'].values,
        'x': x,
        'y': y,
        'altitude': alt_m,
    })


def main():
    logger = configurar_logger('03_proyeccion')
    logger.info("=" * 60)
    logger.info(f" PROYECCIÓN {config.CRS_ORIGEN} → {config.CRS_DESTINO}")
    logger.info("=" * 60)

    with cronometrar(logger, "carga trayectorias limpias"):
        df = pd.read_parquet(config.F_LIMPIO)
    logger.info(f"Cargados {len(df):,} puntos de {df['flight_id'].nunique():,} vuelos")

    logger.info(f"Coordenadas originales (WGS84):")
    logger.info(f"  Latitud:  [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}] grados")
    logger.info(f"  Longitud: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}] grados")
    logger.info(f"  Altitud:  [{df['altitude'].min():.0f}, {df['altitude'].max():.0f}] pies")

    with cronometrar(logger, "proyección LCC + conversión de altitud"):
        transformer = Transformer.from_crs(
            config.CRS_ORIGEN, config.CRS_DESTINO, always_xy=True
        )
        df_proy = proyectar(df, transformer)

    logger.info(f"Coordenadas proyectadas (LCC):")
    logger.info(f"  X: [{df_proy['x'].min():,.0f}, {df_proy['x'].max():,.0f}] m")
    logger.info(f"  Y: [{df_proy['y'].min():,.0f}, {df_proy['y'].max():,.0f}] m")
    logger.info(f"  Altitud: [{df_proy['altitude'].min():.0f}, {df_proy['altitude'].max():.0f}] m")

    n_nan = df_proy[['x', 'y', 'altitude']].isna().sum().sum()
    if n_nan:
        logger.warning(f"AVISO: {n_nan} valores NaN tras la proyección")
    else:
        logger.info("Sin NaN tras la proyección")

    with cronometrar(logger, "escritura trayectorias proyectadas"):
        df_proy.to_parquet(config.F_PROYECTADO, index=False)
    logger.info(f"Guardado en: {config.F_PROYECTADO}")


if __name__ == "__main__":
    main()