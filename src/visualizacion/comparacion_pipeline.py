import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
FLIGHT_ID = 248750938
RUTA_CRUDO = "../../datos/2022-01-01.parquet"
RUTA_LIMPIO = "../../resultados/trayectorias_limpias.parquet"
RUTA_REMUESTREADO = "../../resultados/trayectorias_remuestreadas.parquet"

# === CARGA ===
columnas = ['flight_id', 'timestamp', 'latitude', 'longitude', 'altitude']

df_crudo = pd.read_parquet(RUTA_CRUDO, columns=columnas)
df_crudo = df_crudo[df_crudo['flight_id'] == FLIGHT_ID].sort_values('timestamp')

df_limpio = pd.read_parquet(RUTA_LIMPIO, columns=columnas)
df_limpio = df_limpio[df_limpio['flight_id'] == FLIGHT_ID].sort_values('timestamp')

df_remuestreado = pd.read_parquet(RUTA_REMUESTREADO, columns=columnas)
df_remuestreado = df_remuestreado[df_remuestreado['flight_id'] == FLIGHT_ID].sort_values('timestamp')

print(f"Vuelo {FLIGHT_ID}:")
print(f"  Crudo:        {len(df_crudo)} puntos")
print(f"  Limpio:       {len(df_limpio)} puntos")
print(f"  Remuestreado: {len(df_remuestreado)} puntos")

# === GRÁFICAS ===
fig, axs = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle(f"Comparación pipeline — Vuelo {FLIGHT_ID}", fontsize=16)

# --- FILA 1: DATOS CRUDOS ---
axs[0, 0].set_title(f"Crudo — Trayectoria XY ({len(df_crudo)} pts)")
axs[0, 0].plot(df_crudo['longitude'], df_crudo['latitude'], 'r-', lw=0.5)
axs[0, 0].scatter(df_crudo['longitude'], df_crudo['latitude'], s=1, c='black')
axs[0, 0].set_xlabel("Longitud")
axs[0, 0].set_ylabel("Latitud")

axs[0, 1].set_title("Crudo — Perfil Altitud")
axs[0, 1].plot(df_crudo['altitude'].values, 'r-', lw=0.5)
axs[0, 1].scatter(range(len(df_crudo)), df_crudo['altitude'].values, s=1, c='black')
axs[0, 1].set_ylabel("Altitud (pies)")

lat_diff_c = np.log(np.abs(np.diff(df_crudo['latitude'].values)) + 1)
lon_diff_c = np.log(np.abs(np.diff(df_crudo['longitude'].values)) + 1)
axs[0, 2].set_title("Crudo — Log-Diff")
axs[0, 2].scatter(lon_diff_c, lat_diff_c, s=1, c='black')
axs[0, 2].set_xlabel("Cambio Longitud")
axs[0, 2].set_ylabel("Cambio Latitud")

# --- FILA 2: DATOS LIMPIOS (tras los 5 filtros) ---
axs[1, 0].set_title(f"Limpio — Trayectoria XY ({len(df_limpio)} pts)")
axs[1, 0].plot(df_limpio['longitude'], df_limpio['latitude'], 'g-', lw=0.6)
axs[1, 0].scatter(df_limpio['longitude'], df_limpio['latitude'], s=1, c='black')
axs[1, 0].set_xlabel("Longitud")
axs[1, 0].set_ylabel("Latitud")

axs[1, 1].set_title("Limpio — Perfil Altitud")
axs[1, 1].plot(df_limpio['altitude'].values, 'g-', lw=0.6)
axs[1, 1].scatter(range(len(df_limpio)), df_limpio['altitude'].values, s=1, c='black')
axs[1, 1].set_ylabel("Altitud (pies)")

lat_diff_l = np.log(np.abs(np.diff(df_limpio['latitude'].values)) + 1)
lon_diff_l = np.log(np.abs(np.diff(df_limpio['longitude'].values)) + 1)
axs[1, 2].set_title("Limpio — Log-Diff")
axs[1, 2].scatter(lon_diff_l, lat_diff_l, s=1, c='black')
axs[1, 2].set_xlabel("Cambio Longitud")
axs[1, 2].set_ylabel("Cambio Latitud")

# --- FILA 3: DATOS REMUESTREADOS ---
axs[2, 0].set_title(f"Remuestreado — Trayectoria XY ({len(df_remuestreado)} pts)")
axs[2, 0].plot(df_remuestreado['longitude'], df_remuestreado['latitude'], 'b-', lw=0.8)
axs[2, 0].scatter(df_remuestreado['longitude'], df_remuestreado['latitude'], s=3, c='black')
axs[2, 0].set_xlabel("Longitud")
axs[2, 0].set_ylabel("Latitud")

axs[2, 1].set_title("Remuestreado — Perfil Altitud")
axs[2, 1].plot(df_remuestreado['altitude'].values, 'b-', lw=0.8)
axs[2, 1].scatter(range(len(df_remuestreado)), df_remuestreado['altitude'].values, s=3, c='black')
axs[2, 1].set_ylabel("Altitud (pies)")

lat_diff_r = np.log(np.abs(np.diff(df_remuestreado['latitude'].values)) + 1)
lon_diff_r = np.log(np.abs(np.diff(df_remuestreado['longitude'].values)) + 1)
axs[2, 2].set_title("Remuestreado — Log-Diff")
axs[2, 2].scatter(lon_diff_r, lat_diff_r, s=3, c='black')
axs[2, 2].set_xlabel("Cambio Longitud")
axs[2, 2].set_ylabel("Cambio Latitud")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(top=0.94, hspace=0.35)
plt.show()