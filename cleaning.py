import pandas as pd
import numpy as np

from pathlib import Path

"""
    DATASET 1: Dry Bean Dataset
    El primer dataset no requiere de mucho preprocesamiento, ya que no tiene valores nulos
    y todos los features son numéricos. Tampoco contiene outliers.
    Solo hace falta eliminar duplicados.
    ___ Mas info en el readme ___
"""

def load_data(file_name, header=0, index_col=None):
    ROOT = Path(__file__).resolve().parent
    # le agrego argumentos extra porque tengo otros datasets que requieren cargarse distinto
    data = pd.read_csv(ROOT / "data" / file_name, header=header, index_col=index_col)
    return data

# load dataset
bean = load_data("Dry_Bean_Dataset.csv")
bean = bean.drop_duplicates()
print(bean.head())
# como el dataset originalmente está ordenado por clases, lo randomizo
bean = bean.sample(frac=1, random_state=67).reset_index(drop=True)


"""
    DATASET 2: TRACKS
    Para el segundo dataset, se requirio de mucho preprocesamiento,
    ___ Mas info en el readme ___
"""

# load data
tracks = load_data("tracks.csv", header=[0, 1], index_col=0)

# me quedo con las columnas que me importan para mi dataframe
# maneja doble header
keep_columns = [("album", "date_released"),
                ("album", "favorites"),
                ("album", "tracks"),
                ("album", "type" ),
                ("album", "listens"),
                ("artist", "favorites"),
                ("track", "bit_rate"),
                ("track", "date_created"),
                ("track", "duration"),
                ("track", "favorites"),
                ("track", "genre_top"),
                ("track", "interest"),
                ("track", "number"),
                ]
tracks = tracks[keep_columns]

# como el objetivo es predecir generos, voy a eliminar todos los nan en la columna genre_top
# esta sera la columna target 
tracks = tracks.dropna(subset=[("track", "genre_top")])

# para todas las columnas de date voy a convertir el formato fecha a int
for col in tracks.columns:
    if "date" in col[1]:
        tracks[col] = pd.to_datetime(tracks[col], errors="coerce").astype("int64")

# para album type primero imputo con unknown para no perder los nan
tracks[("album", "type")] = tracks[("album", "type")].fillna("unknown")

# para no tener problemas con el modelo, paso todas las columnas de 2 headers a 1
tracks.columns = tracks.columns.map("_".join).str.strip("_")

# voy a hacerlo dummies porque no son tantas categorias
tracks = pd.get_dummies(tracks, columns=["album_type"], prefix="album_type", dtype=int)

# ahora si puedo borrar todos los nan
tracks = tracks.dropna()

print(tracks.head())