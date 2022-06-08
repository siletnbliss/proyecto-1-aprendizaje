import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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
np.random.seed(42)
wine_data = load_data()
X = wine_data.drop("quality", axis = 1)
y = wine_data.get("quality").to_numpy()

correlation = wine_data.corr()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 2.Exploracion de datos

# Distribucion
wine_data.hist(bins=50, figsize=(8, 8))
save_fig("wine-histogram")
plt.show()

# Escala y rango de valores

# Correlacion
plt.matshow(correlation, cmap = "inferno")
plt.xticks(range(wine_data.select_dtypes(["number"]).shape[1]), wine_data.select_dtypes(["number"]).columns, rotation = 90)
plt.yticks(range(wine_data.select_dtypes(["number"]).shape[1]), wine_data.select_dtypes(["number"]).columns)
plt.colorbar()
save_fig("wine-correlation")
plt.show()

# 3.Preparacion de datos

# Escalar
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])
wine_data_tr = num_pipeline.fit_transform(wine_data)
X_test_tr = num_pipeline.fit_transform(X_test)

# 4.Exploracion de modelos (3 modelos distintos)
score_list = []

def cross_validation(model, title):
    scores = cross_val_score(model, X_test_tr, y_test, cv=2)
    acc = scores.mean()
    print("%s accuracy: %0.4f (+/- %0.4f)" % (title, acc, scores.std() * 2))
    score_list.append({"score": acc, "name": title})
    return acc

print(X_test_tr)
print(y_test)

# KNearest Classification
knclf = KNeighborsClassifier()
knclf.fit(X_train, y_train)
cross_validation(knclf, "KN")

# Decision Tree
dtclf = DecisionTreeClassifier()
dtclf.fit(X_train, y_train)
cross_validation(dtclf, "Decision Tree")

# Random Forest
rfclf = RandomForestClassifier()
rfclf.fit(X_train, y_train)
cross_validation(rfclf, "Random Forest")

# Gradient
le = LabelEncoder()
xgclf = XGBClassifier()
xgclf.fit(X_train, le.fit_transform(y_train))
cross_validation(xgclf, "Gradient")

print(score_list)

# 5.Afinacion de modelos


# 6.Solucion
