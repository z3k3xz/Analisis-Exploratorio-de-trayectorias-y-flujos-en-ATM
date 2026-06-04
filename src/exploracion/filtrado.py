import pandas as pd
 
# IDS FILTRADO POR LLEGADA Y SALIDA
df_meta = pd.read_csv("../../datos/flight_list.csv")
vuelos_interes = df_meta[(df_meta['adep'] == 'EGLL') & (df_meta['ades'] == 'EIDW')]
ids_filtrados = vuelos_interes['flight_id'].unique()
 
# GUARDAR LOS IDS ENCONTRADOS EN UN .TXT
with open("../../resultados/ids_seleccionados.txt", "w") as f:
    for flight_id in ids_filtrados:
        f.write(f"{flight_id}\n")
 
print(f"Hecho: Se han guardado {len(ids_filtrados)} IDs en 'resultados/ids_seleccionados.txt'")


