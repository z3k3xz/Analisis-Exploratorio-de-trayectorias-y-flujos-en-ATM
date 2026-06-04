import pandas as pd

# 1. Cargamos el archivo 
df = pd.read_parquet("../../datos/parquet/2022-01-01.parquet", engine='pyarrow')

# 2. Preparamos el control
total_filas = len(df)
print("--- CONTROL: Presiona [ENTER] para ver la siguiente línea. Escribe 'q' para salir. ---")

# 3. Bucle línea a línea
for i in range(total_filas):
    # Extraemos la fila actual
    fila = df.iloc[i]
    
    # La imprimimos de forma limpia
    print(f"\n>>> LÍNEA {i} <<<")
    print(fila)
    
    # Pausamos y esperamos tu orden
    entrada = input("\n[Presiona Enter para la siguiente...]")
    
    if entrada.lower() == 'q':
        print("Lectura finalizada.")
        break