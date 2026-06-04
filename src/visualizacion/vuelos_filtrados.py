import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. CARGAR SOLO LOS IDs SELECCIONADOS

path = "../../resultados/ids_seleccionados.txt"
path_ruidos = "../../resultados/vuelos_ruidosos.txt"

if not os.path.exists(path_ruidos):
    print("Error: No se encuentra 'vuelos_ruidosos.txt'. Ejecuta primero ruidos.py")
    exit()

with open(path_ruidos, "r") as f:
    ids_proyecto = [int(line.split(',')[0]) for line in f if line.strip()]

print(f"Cargados {len(ids_proyecto)} IDs de vuelos")

# 2. CARGAR DATASET
ruta_parquet = "../../datos/2022-01-01.parquet"
columnas = ['flight_id', 'latitude', 'longitude', 'altitude', 'timestamp']

df_completo = pd.read_parquet(ruta_parquet, columns=columnas)

# FILTRO MAESTRO: Solo nos quedamos con los puntos de los IDs del TXT
df = df_completo[df_completo['flight_id'].isin(ids_proyecto)]
unique_flights = df['flight_id'].unique()

print(f"Análisis listo para {len(unique_flights)} vuelos de la ruta seleccionada.")

# 3. BUCLE PARA LOS VUELOS FILTRADOS
for i in range(len(unique_flights)):
    # Obtenemos el perfil del vuelo actual
    profile = df[df['flight_id'] == unique_flights[i]].sort_values('timestamp')
    
    lat = profile['latitude'].values
    lon = profile['longitude'].values
    alt = profile['altitude'].values
    n = len(profile)

    # Lógica de cálculo de diferencias (Detector de errores)
    lat_diff = np.log(np.abs(np.diff(lat)) + 1)
    lon_diff = np.log(np.abs(np.diff(lon)) + 1)

    # Creación del panel de gráficas (Dashboard) 
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Análisis de Flujo ATM - ID: {unique_flights[i]}", fontsize=16)

    # --- FILA SUPERIOR ---
    axs[0, 0].plot(lat, color='red', lw=0.5)
    axs[0, 0].scatter(range(n), lat, s=1, color='black')
    axs[0, 0].set_title("Perfil Latitud (N-S)")

    axs[0, 1].plot(lon, color='red', lw=0.5)
    axs[0, 1].scatter(range(n), lon, s=1, color='black')
    axs[0, 1].set_title("Perfil Longitud (E-O)")

    axs[0, 2].plot(lon, lat, color='blue', lw=0.8) 
    axs[0, 2].scatter(lon, lat, s=1, color='black')
    axs[0, 2].set_xlabel("Longitud")
    axs[0, 2].set_ylabel("Latitud")
    axs[0, 2].set_title("Mapa de Ruta Real")

    # --- FILA INFERIOR ---
    jitter_lat = lat_diff + np.random.normal(0, 0.001, len(lat_diff))
    jitter_lon = lon_diff + np.random.normal(0, 0.001, len(lon_diff))
    
    axs[1, 0].plot(jitter_lon, jitter_lat, color='red', alpha=0.2, lw=0.5)
    axs[1, 0].scatter(jitter_lon, jitter_lat, s=1, color='black')
    axs[1, 0].set_title("Detector de Errores Sensor")

    axs[1, 1].plot(alt, color='green', lw=0.5)
    axs[1, 1].scatter(range(n), alt, s=1, color='black')
    axs[1, 1].set_title("Perfil Altitud (Pies)")

    axs[1, 2].axis('off') 

    plt.tight_layout()
    plt.show()

    input(f"Vuelo {i+1}/{len(unique_flights)}. Presiona ENTER para el siguiente...")
    plt.close()