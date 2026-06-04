import pandas as pd
import matplotlib.pyplot as plt

ruta_parquet = "../../resultados/preparacion/datos_fusionados.parquet"

print("Leyendo altitudes...")
df_alt = pd.read_parquet(ruta_parquet, columns=['altitude'])

plt.figure(figsize=(12, 6))
plt.hist(df_alt[df_alt['altitude'] > 1000]['altitude'], bins=100, color='skyblue', edgecolor='black')

plt.title("Distribución Vertical del Tráfico Aéreo (Niveles de Vuelo)")
plt.xlabel("Altitud (Pies)")
plt.ylabel("Número de registros (puntos de trayectoria)")
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("histograma_altitud.png", dpi=200)
print("Guardado: histograma_altitud.png")