import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "wines/wines.csv"

# Directorio para graficos
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok = True)

# Guardar grafico
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def load_data(data_path = DATA_PATH):
    return pd.read_csv(data_path)

# 1.Generacion del conjunto de prueba
wine_data = load_data()
wine_data.hist(bins=20, figsize=(10, 10))
save_fig("histogram")
plt.show()

# 2.Exploracion de datos


# 3.Preparacion de datos

# 4.Exploracion de modelos (3 modelos distintos)

# 5.Afinacion de modelos

# 6.Solucion
