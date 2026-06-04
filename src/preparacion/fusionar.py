
from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.utils import configurar_logger, cronometrar


def fusionar(ficheros: list[str], logger) -> pd.DataFrame:
    """Lee la lista de parquets indicada y concatena en un único DataFrame."""
    dfs = []
    for nombre in ficheros:
        ruta = config.DIR_PARQUETS / nombre
        if not ruta.exists():
            logger.error(f"Falta el fichero: {ruta}")
            raise FileNotFoundError(ruta)
        df = pd.read_parquet(ruta, columns=config.COLUMNAS_TRAYECTORIAS)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        logger.info(f"  {nombre}: {df['flight_id'].nunique():,} vuelos, "
                    f"{len(df):,} puntos")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ficheros', nargs='+', default=config.PATRON_PARQUETS_SEMANA,
        help='Nombres de los parquets diarios a fusionar (sin ruta).'
    )
    args = parser.parse_args()

    logger = configurar_logger('01_fusionar')
    logger.info("=" * 60)
    logger.info(" FUSIÓN DE PARQUETS DIARIOS")
    logger.info("=" * 60)
    logger.info(f"Ventana: {config.FECHA_INICIO} a {config.FECHA_FIN}")
    logger.info(f"Ficheros previstos: {len(args.ficheros)}")

    with cronometrar(logger, "fusión bruta"):
        df = fusionar(args.ficheros, logger)

    logger.info(f"\nTotal antes de cruzar metadatos:")
    logger.info(f"  Vuelos: {df['flight_id'].nunique():,}")
    logger.info(f"  Puntos: {len(df):,}")
    logger.info(f"  Rango temporal: {df['timestamp'].min()} → {df['timestamp'].max()}")

    with cronometrar(logger, "cruce con flight_list.csv"):
        df_meta = pd.read_csv(config.RUTA_METADATOS)
        ids_presentes = df['flight_id'].unique()
        ids_con_meta = set(df_meta[df_meta['flight_id'].isin(ids_presentes)]['flight_id'])
        ids_sin_meta = set(ids_presentes) - ids_con_meta

    n_antes = df['flight_id'].nunique()
    df = df[df['flight_id'].isin(ids_con_meta)]
    n_despues = df['flight_id'].nunique()

    logger.info(f"\nResultado tras filtrado por metadatos:")
    logger.info(f"  Vuelos sin metadatos descartados: {len(ids_sin_meta):,}")
    logger.info(f"  Vuelos retenidos: {n_antes:,} → {n_despues:,}")
    logger.info(f"  Puntos retenidos: {len(df):,}")

    with cronometrar(logger, "escritura parquet fusionado"):
        df.to_parquet(config.F_FUSIONADO, index=False)

    logger.info(f"Guardado en: {config.F_FUSIONADO}")


if __name__ == "__main__":
    main()