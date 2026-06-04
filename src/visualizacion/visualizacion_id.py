import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- USO: python visualizacion_id_export.py <flight_id> [crudo|limpio] ---
if len(sys.argv) < 2:
    print("Uso: python visualizacion_id_export.py <flight_id> [crudo|limpio]")
    exit(1)

FLIGHT_ID = int(sys.argv[1])
modo = sys.argv[2] if len(sys.argv) > 2 else "crudo"

if modo == "crudo":
    ruta = "../../resultados/preparacion/datos_fusionados.parquet"
else:
    ruta = "../../resultados/preparacion/trayectorias_limpias.parquet"

columnas = ['flight_id', 'timestamp', 'latitude', 'longitude', 'altitude']
df = pd.read_parquet(ruta, columns=columnas)
profile = df[df['flight_id'] == FLIGHT_ID].sort_values('timestamp')

if len(profile) == 0:
    print(f"ERROR: flight_id {FLIGHT_ID} no encontrado en {ruta}")
    exit(1)

lat = profile['latitude'].values
lon = profile['longitude'].values
alt = profile['altitude'].values
n = len(profile)

lat_diff = np.log(np.abs(np.diff(lat)) + 1)
lon_diff = np.log(np.abs(np.diff(lon)) + 1)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"Diagnóstico de Vuelo — ID: {FLIGHT_ID} ({modo})", fontsize=16)

axs[0, 0].plot(lat, color='red', lw=0.5)
axs[0, 0].scatter(range(n), lat, s=1, color='black')
axs[0, 0].set_title("Perfil Latitud (N-S)")

axs[0, 1].plot(lon, color='red', lw=0.5)
axs[0, 1].scatter(range(n), lon, s=1, color='black')
axs[0, 1].set_title("Perfil Longitud (E-O)")

axs[0, 2].plot(lon, lat, color='red', lw=0.8)
axs[0, 2].scatter(lon, lat, s=1, color='black')
axs[0, 2].set_xlabel("Longitud")
axs[0, 2].set_ylabel("Latitud")
axs[0, 2].set_title("Trayectoria XY")

jitter_lat = lat_diff + np.random.normal(0, 0.001, len(lat_diff))
jitter_lon = lon_diff + np.random.normal(0, 0.001, len(lon_diff))
axs[1, 0].plot(jitter_lon, jitter_lat, color='red', alpha=0.2, lw=0.5)
axs[1, 0].scatter(jitter_lon, jitter_lat, s=1, color='black')
axs[1, 0].set_xlabel("Cambio Longitud (log)")
axs[1, 0].set_ylabel("Cambio Latitud (log)")
axs[1, 0].set_title("Detector de Errores (Log-Diff)")

axs[1, 1].plot(alt, color='red', lw=0.5)
axs[1, 1].scatter(range(n), alt, s=1, color='black')
axs[1, 1].set_title("Perfil Altitud (Pies)")

axs[1, 2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(top=0.94, hspace=0.35)

nombre = f"diagnostico_{FLIGHT_ID}_{modo}.png"
plt.savefig(nombre, dpi=200)
print(f"Guardado: {nombre}")