import pandas as pd
import numpy as np

# === CONFIGURACIÓN ===
RUTA_FUSIONADO = "../../resultados/preparacion/datos_fusionados.parquet"
RUTA_LIMPIAS = "../../resultados/preparacion/trayectorias_limpias.parquet"
RUTA_METADATOS = "../../datos/flight_list.csv"

# Umbral de hueco (debe coincidir con limpieza.py)
UMBRAL_HUECO = 300  # segundos

print("Cargando datos...")
df_fusionado = pd.read_parquet(RUTA_FUSIONADO, columns=['flight_id', 'timestamp'])
df_limpias = pd.read_parquet(RUTA_LIMPIAS, columns=['flight_id'])
df_meta = pd.read_csv(RUTA_METADATOS)

ids_fusionado = set(df_fusionado['flight_id'].unique())
ids_limpias = set(df_limpias['flight_id'].unique())
ids_descartados = ids_fusionado - ids_limpias

print(f"Vuelos fusionados: {len(ids_fusionado)}")
print(f"Vuelos tras limpieza: {len(ids_limpias)}")
print(f"Vuelos descartados: {len(ids_descartados)}")

print(f"\nIdentificando vuelos descartados por hueco > {UMBRAL_HUECO}s...")
descartados_hueco = []

for fid in ids_descartados:
    vuelo = df_fusionado[df_fusionado['flight_id'] == fid].sort_values('timestamp')
    if len(vuelo) < 2:
        continue
    diffs = vuelo['timestamp'].diff().dt.total_seconds().dropna()
    max_hueco = diffs.max()
    if max_hueco > UMBRAL_HUECO:
        descartados_hueco.append({'flight_id': fid, 'max_hueco_s': max_hueco})

df_hueco = pd.DataFrame(descartados_hueco)
print(f"Vuelos descartados por hueco: {len(df_hueco)}")

df_hueco = df_hueco.merge(df_meta[['flight_id', 'adep', 'name_adep', 'ades', 'name_ades']], on='flight_id', how='left')
df_hueco['ruta'] = df_hueco['adep'].fillna('?') + ' → ' + df_hueco['ades'].fillna('?')

# 10 rutas más afectadas 
conteo = df_hueco['ruta'].value_counts().head(10)

print(f"\n{'=' * 60}")
print(f" TOP 10 RUTAS MÁS AFECTADAS POR HUECO > {UMBRAL_HUECO}s")
print(f"{'=' * 60}")
for i, (ruta, count) in enumerate(conteo.items(), 1):
    fila = df_hueco[df_hueco['ruta'] == ruta].iloc[0]
    name_adep = fila.get('name_adep', '')
    name_ades = fila.get('name_ades', '')
    adep = fila.get('adep', '?')
    ades = fila.get('ades', '?')

    ids_ruta_total = df_meta[(df_meta['adep'] == adep) & (df_meta['ades'] == ades) &
                              (df_meta['flight_id'].isin(ids_fusionado))]['flight_id'].nunique()
    pct = count / ids_ruta_total * 100 if ids_ruta_total > 0 else 0

    print(f"  {i:2d}. {adep} ({name_adep}) → {ades} ({name_ades})")
    print(f"      Descartados: {count} de {ids_ruta_total} ({pct:.0f}%)")