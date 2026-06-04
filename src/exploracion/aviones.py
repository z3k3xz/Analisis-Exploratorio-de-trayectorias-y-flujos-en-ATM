import pandas as pd

# 1. Configuración del nombre del archivo
archivo_csv = "../../datos/flight_list.csv"

# 2. Cargamos el CSV completo 
df = pd.read_csv(archivo_csv)

total_filas = len(df)
print("--- Presiona [ENTER] para la siguiente línea | Escribe 'q' para salir ---")

# 3. Bucle de lectura línea a línea
for i in range(total_filas):
    fila = df.iloc[i]
    
    # Imprimimos los datos
    print(f"\n>>> LÍNEA {i} <<<")
    print(fila)
    
    # Pausa manual
    entrada = input("\n[Siguiente...]")
    
    # Opción para salir rápido
    if entrada.lower() == 'q':
        break

print("Fin de la lectura.")