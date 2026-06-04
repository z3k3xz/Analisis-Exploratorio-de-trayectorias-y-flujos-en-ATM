import pandas as pd
import numpy as np

ruta_parquet = "../../resultados/preparacion/datos_fusionados.parquet"
df = pd.read_parquet(ruta_parquet, columns=['flight_id', 'latitude', 'longitude'])

def calcular_puntuacion_ruido(group):
    diff_lat = np.abs(np.diff(group['latitude']))
    diff_lon = np.abs(np.diff(group['longitude']))
    return np.std(diff_lat) + np.std(diff_lon)

vuelos_unicos = df['flight_id'].unique()[:500]
df_muestra = df[df['flight_id'].isin(vuelos_unicos)]

puntuaciones = df_muestra.groupby('flight_id').apply(calcular_puntuacion_ruido)

top_ruido = puntuaciones.sort_values(ascending=False).head(10)

print("\n--- VUELOS CON MÁS RUIDO ---")
print(top_ruido)
print("\nUsa estos IDs en visualizacion_id_export.py")