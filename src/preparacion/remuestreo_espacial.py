
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import configurar_logger, cronometrar, remuestrear_trayectoria


def remuestrear_vuelo(grupo: pd.DataFrame, n_puntos: int):
    """Wrapper sobre utils.remuestrear_trayectoria para un DataFrame."""
    grupo = grupo.sort_values('timestamp')
    resultado = remuestrear_trayectoria(
        grupo['x'].values, grupo['y'].values, grupo['altitude'].values,
        n_puntos=n_puntos,
    )
    if resultado is None:
        return None
    x_i, y_i, alt_i = resultado
    return pd.DataFrame({
        'flight_id': grupo['flight_id'].iloc[0],
        'point_index': np.arange(n_puntos),
        'x': x_i,
        'y': y_i,
        'altitude': alt_i,
    })


def estadisticas_distancia(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la distancia 3D total recorrida por cada vuelo, en km."""
    filas = []
    for fid, grupo in df.groupby('flight_id'):
        grupo = grupo.sort_values('timestamp')
        dx = np.diff(grupo['x'].values)
        dy = np.diff(grupo['y'].values)
        dalt = np.diff(grupo['altitude'].values)
        dist = float(np.sum(np.sqrt(dx ** 2 + dy ** 2 + dalt ** 2)))
        filas.append({'flight_id': fid, 'distancia_total_km': dist / 1000})
    return pd.DataFrame(filas)


def main():
    logger = configurar_logger('04_remuestreo')
    logger.info("=" * 60)
    logger.info(f" REMUESTREO ESPACIAL A {config.N_PUNTOS} PUNTOS")
    logger.info("=" * 60)

    with cronometrar(logger, "carga trayectorias proyectadas"):
        df = pd.read_parquet(config.F_PROYECTADO)
    n_vuelos = df['flight_id'].nunique()
    logger.info(f"Cargados {len(df):,} puntos de {n_vuelos:,} vuelos")

    with cronometrar(logger, "estadísticas de distancia recorrida"):
        df_dist = estadisticas_distancia(df)
    logger.info(f"Distancia recorrida (km):")
    logger.info(f"  Mín:     {df_dist['distancia_total_km'].min():,.1f}")
    logger.info(f"  Mediana: {df_dist['distancia_total_km'].median():,.1f}")
    logger.info(f"  Máx:     {df_dist['distancia_total_km'].max():,.1f}")
    logger.info(f"  P99:     {df_dist['distancia_total_km'].quantile(0.99):,.1f}")

    # Marca de calidad: vuelos físicamente imposibles
    n_imposibles = (df_dist['distancia_total_km'] > 20_000).sum()
    if n_imposibles:
        logger.warning(f"AVISO: {n_imposibles} vuelo(s) con distancia recorrida "
                       f"> 20.000 km. Esto excede la mitad de la circunferencia "
                       f"terrestre y puede indicar fallo de proyección o filtro "
                       f"insuficiente. Revisar con diagnostico_vuelos_largos.py.")

    with cronometrar(logger, "remuestreo equidistante"):
        resultados = []
        descartados = 0
        for _, grupo in df.groupby('flight_id'):
            res = remuestrear_vuelo(grupo, config.N_PUNTOS)
            if res is None:
                descartados += 1
            else:
                resultados.append(res)
        df_out = pd.concat(resultados, ignore_index=True)

    n_final = df_out['flight_id'].nunique()
    logger.info(f"Vuelos remuestreados: {n_vuelos:,} → {n_final:,} "
                f"(descartados {descartados:,})")
    logger.info(f"Matriz resultante: {n_final:,} x {config.N_PUNTOS} x 3 "
                f"= vector de {config.N_PUNTOS * 3} componentes por vuelo")

    with cronometrar(logger, "escritura trayectorias normalizadas"):
        df_out.to_parquet(config.F_NORMALIZADO, index=False)
    logger.info(f"Guardado en: {config.F_NORMALIZADO}")


if __name__ == "__main__":
    main()