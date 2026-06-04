
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from pyproj import Transformer

from src import config
from src.utils import configurar_logger, cronometrar


# Radio en metros para considerar que una trayectoria "pasa por" un fix.
# 9 NM ≈ 16,7 km es la distancia operacional típica para considerar
# que un vuelo ha cruzado o sobrevolado un IAF (área de holding).
RADIO_PROXIMIDAD = 16_700


METERING_FIXES = {
    "LOWW": {
        # AIP Austro Control, AD 2.24 STAR. Cuatro IAFs cardinales de LOWW.
        "NERDU": {"lat": 48 + 28/60 + 54/3600, "lon": 16 + 6/60 + 0/3600,
                  "sector": "NW", "stars": "MIKOV, REKLU, TOVKA"},
        "MABOD": {"lat": 48 + 34/60 + 28/3600, "lon": 16 + 41/60 + 24/3600,
                  "sector": "NE", "stars": "NATEX, LAPNA, NIGSI, OBUTI"},
        "BALAD": {"lat": 47 + 46/60 +  0/3600, "lon": 16 + 14/60 +  0/3600,
                  "sector": "SW", "stars": "ABTAN, BARUG, NEMAL"},
        "PESAT": {"lat": 47 + 42/60 + 54/3600, "lon": 17 +  3/60 + 12/3600,
                  "sector": "SE", "stars": "BUDEX, LANUX, MASUR, VENEN"},
    },
    "EKCH": {
        # AIP Naviair (verificados en OpenNav). Sólo dos de los cuatro IAFs;
        # los otros dos (ERNOV, TIDVU) no se incluyen por inconsistencia de
        # fuentes — ver documentación de la sección 5.4 de la memoria.
        "LUGAS": {"lat": 55 + 19/60 + 47/3600, "lon": 10 + 57/60 + 47/3600,
                  "sector": "SW", "stars": "STAR vía LUGAS"},
        "ROSBI": {"lat": 55 + 50/60 + 58/3600, "lon": 10 + 55/60 + 55/3600,
                  "sector": "NW", "stars": "STAR vía ROSBI"},
    },
}


def proyectar_fixes(icao: str, transformer: Transformer) -> dict[str, dict]:
    """Devuelve los metering fixes con coordenadas LCC añadidas."""
    fixes = METERING_FIXES[icao].copy()
    out = {}
    for nombre, data in fixes.items():
        x, y = transformer.transform(data['lon'], data['lat'])
        out[nombre] = {**data, 'x': float(x), 'y': float(y)}
    return out


def calcular_proximidad_cluster(df_tray: pd.DataFrame, vuelos: np.ndarray,
                                fix_x: float, fix_y: float,
                                radio: float) -> dict:
    """
    Para los vuelos indicados, cuenta cuántos pasan a menos de radio
    de un punto (fix_x, fix_y). Un vuelo "pasa por" el fix si al menos
    uno de sus puntos remuestreados está dentro del radio.

    Devuelve también la distancia mínima mediana, útil para ver si los
    vuelos del cluster están alineados con el fix de forma sistemática
    o sólo lo cruzan por casualidad.
    """
    df = df_tray[df_tray['flight_id'].isin(vuelos)].copy()
    df['dist_fix'] = np.sqrt((df['x'] - fix_x) ** 2 + (df['y'] - fix_y) ** 2)
    dist_min_por_vuelo = df.groupby('flight_id')['dist_fix'].min()

    n_total = len(dist_min_por_vuelo)
    n_cerca = int((dist_min_por_vuelo <= radio).sum())
    pct_cerca = (n_cerca / n_total * 100) if n_total > 0 else 0.0
    return {
        'n_total': n_total,
        'n_cerca': n_cerca,
        'pct_cerca': pct_cerca,
        'dist_min_mediana_km': float(dist_min_por_vuelo.median() / 1000),
    }


def procesar_aeropuerto(icao: str, esquema: str, logger) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info(f" VALIDACIÓN CON METERING FIXES — {icao} — esquema: {esquema}")
    logger.info("=" * 60)

    if icao not in METERING_FIXES:
        logger.warning(f"No hay metering fixes declarados para {icao}. Se omite.")
        return

    ruta_tray = config.f_micro_trayectorias(icao)
    ruta_clusters = config.f_micro_clusters(icao, esquema)
    if not (ruta_tray.exists() and ruta_clusters.exists()):
        logger.error(f"Faltan ficheros para {icao}/{esquema}.")
        return

    df_tray = pd.read_parquet(ruta_tray)
    df_clusters = pd.read_parquet(ruta_clusters)

    # Proyectar los fixes al CRS LCC
    transformer = Transformer.from_crs(config.CRS_ORIGEN, config.CRS_DESTINO,
                                   always_xy=True)
    fixes = proyectar_fixes(icao, transformer)
    logger.info(f"Metering fixes para {icao}:")
    for nombre, d in fixes.items():
        logger.info(f"  {nombre} (sector {d['sector']}): "
                    f"WGS84 ({d['lat']:.4f}, {d['lon']:.4f})  →  "
                    f"LCC ({d['x']:,.0f}, {d['y']:,.0f}) m")
    logger.info(f"Radio de proximidad: {RADIO_PROXIMIDAD/1000:.1f} km (≈ 9 NM)")

    clusters_validos = sorted(df_clusters[df_clusters['cluster'] >= 0]['cluster'].unique())
    nombres_fix = list(fixes.keys())

    # Cabecera de la tabla
    cab = f"{'Cluster':<10} {'N':>5}  " + "  ".join(f"{f:>10}" for f in nombres_fix)
    logger.info("")
    logger.info("Porcentaje de vuelos por cluster que pasan a < radio de cada fix:")
    logger.info(cab)
    logger.info("-" * len(cab))

    filas_tabla = []
    for cid in clusters_validos:
        vuelos = df_clusters[df_clusters['cluster'] == cid]['flight_id'].values
        fila = {'icao': icao, 'esquema': esquema, 'cluster': int(cid),
                'n_vuelos': int(len(vuelos))}
        cells = []
        for nombre in nombres_fix:
            d = fixes[nombre]
            prox = calcular_proximidad_cluster(
                df_tray, vuelos, d['x'], d['y'], RADIO_PROXIMIDAD
            )
            fila[f'pct_cerca_{nombre}'] = prox['pct_cerca']
            fila[f'dist_min_km_{nombre}'] = prox['dist_min_mediana_km']
            cells.append(f"{prox['pct_cerca']:>9.1f}%")
        logger.info(f"C{cid:<8d}  {len(vuelos):>5d}  " + "  ".join(cells))
        filas_tabla.append(fila)

    # Asignación dominante: para cada cluster, el fix con mayor % de vuelos cerca
    logger.info("")
    logger.info("Asignación dominante por cluster (fix con mayor % de proximidad):")
    for fila in filas_tabla:
        cid = fila['cluster']
        pcts = {f: fila[f'pct_cerca_{f}'] for f in nombres_fix}
        fix_dom = max(pcts, key=pcts.get)
        pct_dom = pcts[fix_dom]
        if pct_dom >= 50:
            verdict = f"→ {fix_dom} ({pct_dom:.1f}%)"
        elif pct_dom >= 25:
            verdict = f"→ {fix_dom} parcial ({pct_dom:.1f}%)"
        else:
            verdict = "→ no alineado con ningún fix"
        logger.info(f"  Cluster {cid} (N={fila['n_vuelos']}): {verdict}")

    # Guardado de la tabla
    df_tabla = pd.DataFrame(filas_tabla)
    ruta_salida = config.DIR_VALIDACIONES / f"metering_fixes_{icao}_{esquema}.parquet"
    df_tabla.to_parquet(ruta_salida, index=False)
    logger.info("")
    logger.info(f"Tabla guardada en: {ruta_salida}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aeropuertos', nargs='+', default=config.AEROPUERTOS_MICRO)
    parser.add_argument('--esquema', type=str, default=config.ESQUEMA_DEFAULT,
                        choices=config.ESQUEMAS_DISTANCIA)
    args = parser.parse_args()

    logger = configurar_logger(f'12_metering_fixes_{args.esquema}')
    logger.info("=" * 60)
    logger.info(" VALIDACIÓN CUANTITATIVA CON METERING FIXES")
    logger.info("=" * 60)
    logger.info(f"Esquema:     {args.esquema}")
    logger.info(f"Aeropuertos: {args.aeropuertos}")

    for icao in args.aeropuertos:
        procesar_aeropuerto(icao, args.esquema, logger)


if __name__ == "__main__":
    main()