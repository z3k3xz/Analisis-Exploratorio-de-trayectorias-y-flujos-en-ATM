from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import configurar_logger, cronometrar


def caracterizar_cluster(df_tray: pd.DataFrame, df_meta: pd.DataFrame,
                         cluster_id: int, vuelos: np.ndarray) -> dict:
    resumen = {'cluster': int(cluster_id), 'n_vuelos': int(len(vuelos))}

    meta = df_meta[df_meta['flight_id'].isin(vuelos)]

    # Orígenes y destinos
    if 'adep' in meta.columns:
        origs = meta['adep'].value_counts()
        resumen['top_origenes'] = ', '.join([f"{a} ({c})" for a, c in origs.head(5).items()])
        resumen['n_origenes_distintos'] = int(meta['adep'].nunique())
    else:
        resumen['top_origenes'] = 'N/A'
        resumen['n_origenes_distintos'] = 0

    if 'ades' in meta.columns:
        dests = meta['ades'].value_counts()
        resumen['top_destinos'] = ', '.join([f"{a} ({c})" for a, c in dests.head(5).items()])
        resumen['n_destinos_distintos'] = int(meta['ades'].nunique())
    else:
        resumen['top_destinos'] = 'N/A'
        resumen['n_destinos_distintos'] = 0

    # Rutas
    if 'adep' in meta.columns and 'ades' in meta.columns:
        rutas = (meta['adep'].fillna('?') + ' → ' + meta['ades'].fillna('?')).value_counts()
        resumen['top_rutas'] = ', '.join([f"{r} ({c})" for r, c in rutas.head(5).items()])
        resumen['n_rutas_distintas'] = int(len(rutas))
    else:
        resumen['top_rutas'] = 'N/A'
        resumen['n_rutas_distintas'] = 0

    # Trayectoria
    tray = df_tray[df_tray['flight_id'].isin(vuelos)]
    alt_max = tray.groupby('flight_id')['altitude'].max()
    resumen['alt_crucero_media_m'] = float(alt_max.mean())
    resumen['alt_crucero_min_m'] = float(alt_max.min())
    resumen['alt_crucero_max_m'] = float(alt_max.max())

    # Distancia horizontal recorrida (km)
    dist_km = []
    for fid, grupo in tray.groupby('flight_id'):
        grupo = grupo.sort_values('timestamp')
        dx = np.diff(grupo['x'].values)
        dy = np.diff(grupo['y'].values)
        dist_km.append(float(np.sum(np.sqrt(dx ** 2 + dy ** 2))) / 1000)
    if dist_km:
        resumen['dist_media_km'] = float(np.mean(dist_km))
        resumen['dist_min_km'] = float(np.min(dist_km))
        resumen['dist_max_km'] = float(np.max(dist_km))
    else:
        resumen['dist_media_km'] = resumen['dist_min_km'] = resumen['dist_max_km'] = 0.0

    # Hora de salida (primer punto de cada vuelo)
    primer = tray.sort_values('timestamp').groupby('flight_id')['timestamp'].first()
    if pd.api.types.is_datetime64_any_dtype(primer):
        h = primer.dt.hour
        resumen['hora_salida_media'] = float(h.mean())
        resumen['hora_salida_min'] = int(h.min())
        resumen['hora_salida_max'] = int(h.max())
    else:
        resumen['hora_salida_media'] = -1.0
        resumen['hora_salida_min'] = -1
        resumen['hora_salida_max'] = -1

    return resumen


def main():
    logger = configurar_logger('07_caracterizacion_macro')
    logger.info("=" * 60)
    logger.info(" CARACTERIZACIÓN DE CLUSTERS MACRO")
    logger.info("=" * 60)

    with cronometrar(logger, "carga clusters, trayectorias proyectadas y metadatos"):
        df_clusters = pd.read_parquet(config.F_CLUSTERS_MACRO)
        df_tray = pd.read_parquet(config.F_PROYECTADO)
        df_tray['timestamp'] = pd.to_datetime(df_tray['timestamp'], utc=True)
        df_meta = pd.read_csv(config.RUTA_METADATOS)

    logger.info(f"Columnas en flight_list.csv: {list(df_meta.columns)}")

    clusters_validos = sorted(df_clusters[df_clusters['cluster'] >= 0]['cluster'].unique())
    logger.info(f"Caracterizando {len(clusters_validos)} clusters + ruido")
    logger.info("")

    resumenes = []

    for cid in clusters_validos:
        vuelos = df_clusters[df_clusters['cluster'] == cid]['flight_id'].values
        r = caracterizar_cluster(df_tray, df_meta, cid, vuelos)
        resumenes.append(r)

        logger.info(f"--- Cluster {cid} ({r['n_vuelos']} vuelos) ---")
        logger.info(f"  Orígenes top:  {r['top_origenes']}")
        logger.info(f"  Destinos top:  {r['top_destinos']}")
        logger.info(f"  Rutas top:     {r['top_rutas']}")
        logger.info(f"  Alt crucero:   {r['alt_crucero_media_m']:.0f} m "
                    f"(rango {r['alt_crucero_min_m']:.0f}–{r['alt_crucero_max_m']:.0f})")
        logger.info(f"  Distancia:     {r['dist_media_km']:.0f} km "
                    f"(rango {r['dist_min_km']:.0f}–{r['dist_max_km']:.0f})")
        logger.info(f"  Hora salida:   {r['hora_salida_media']:.1f}h "
                    f"(rango {r['hora_salida_min']}–{r['hora_salida_max']})")
        logger.info("")

    # Ruido
    ruido_ids = df_clusters[df_clusters['cluster'] == -1]['flight_id'].values
    if len(ruido_ids):
        r = caracterizar_cluster(df_tray, df_meta, -1, ruido_ids)
        resumenes.append(r)
        logger.info(f"--- RUIDO ({r['n_vuelos']} vuelos) ---")
        logger.info(f"  Orígenes top:  {r['top_origenes']}")
        logger.info(f"  Destinos top:  {r['top_destinos']}")
        logger.info(f"  Rutas top:     {r['top_rutas']}")
        logger.info(f"  Distancia:     {r['dist_media_km']:.0f} km "
                    f"(rango {r['dist_min_km']:.0f}–{r['dist_max_km']:.0f})")
        logger.info("")

    # Tabla consolidada en parquet
    df_resumen = pd.DataFrame(resumenes)
    ruta_salida = config.DIR_MACRO / "caracterizacion_macro.parquet"
    df_resumen.to_parquet(ruta_salida, index=False)
    logger.info(f"Resumen guardado en: {ruta_salida}")


if __name__ == "__main__":
    main()