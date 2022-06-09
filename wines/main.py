import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = "wines/wines.csv"

# Directorio para graficos
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok = True)

# Guardar grafico
def save_fig(fig_id, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    plt.savefig(path, format=fig_extension, dpi=resolution)

def load_data(data_path = DATA_PATH):
    return pd.read_csv(data_path)

# Exploracion de datos

np.random.seed(42)
wine_data = load_data()

# Distribucion
wine_data.hist(bins=50, figsize=(8, 8))
save_fig("wine-histogram")

# Correlacion
plt.matshow(wine_data.corr(), cmap = "inferno")
plt.xticks(range(wine_data.select_dtypes(["number"]).shape[1]), wine_data.select_dtypes(["number"]).columns, rotation = 90)
plt.yticks(range(wine_data.select_dtypes(["number"]).shape[1]), wine_data.select_dtypes(["number"]).columns)
plt.colorbar()
save_fig("wine-correlation")

# Transformar ('superior' si quality >= 6)
bins = (0, 6, 10)
group_names = ['inferior', 'superior']
wine_data['quality'] = pd.cut(wine_data['quality'], bins = bins, labels = group_names)
le = LabelEncoder()
wine_data['quality'] = le.fit_transform(wine_data['quality'])

# Generacion del conjunto de prueba

X = wine_data.drop("quality", axis = 1)
y = wine_data.get("quality").to_numpy()

#X = X.drop("total sulfur dioxide", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Preparacion de datos

# Escalar
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])
X_train_tr = num_pipeline.fit_transform(X_train)

# Exploracion de modelos (3 modelos distintos)
score_list = []

def cross_validation(model, title):
    le = LabelEncoder()
    scores = cross_val_score(model, X_test, y_test, cv=2)
    acc = scores.mean()
    print("%s accuracy: %0.4f (+/- %0.4f)" % (title, acc, scores.std() * 2))
    score_list.append({"score": acc, "name": title})
    return acc

# Decision Tree
dtclf = DecisionTreeClassifier()
dtclf.fit(X_train_tr, y_train)
cross_validation(dtclf, "Decision Tree")

# Random Forest
rfclf = RandomForestClassifier()
rfclf.fit(X_train_tr, y_train)
cross_validation(rfclf, "Random Forest")

# Support Vector Machine
svmclf = SVC()
svmclf.fit(X_train_tr, y_train)
cross_validation(svmclf, "Support Vector Machine")

# Afinacion de modelos

def tune_model(model, param_grid, title):
    grid_search = GridSearchCV(model, param_grid, cv=10, n_jobs=-1,
                           scoring='accuracy')
    grid_search.fit(X_train_tr, y_train)
    print("\n" + title)
    print(grid_search.best_params_)

# Decision Tree
param_grid = {
    'max_depth': [90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12]
}

tune_model(dtclf, param_grid, "Decision Tree")
dtclf_hp = DecisionTreeClassifier(max_depth=90, max_features=2, min_samples_leaf=5, min_samples_split=8)
dtclf_hp.fit(X_train_tr, y_train)
cross_validation(dtclf_hp, "Decision Tree (Tuned)")

# Random Forest
param_grid = {
    'bootstrap': [True],
    'max_depth': [90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [50, 100, 200]
}

tune_model(rfclf, param_grid, "Random Forest")
rfclf_hp = RandomForestClassifier(max_depth=110, max_features=2, min_samples_leaf=3, min_samples_split=8, n_estimators=50)
rfclf_hp.fit(X_train_tr, y_train)
cross_validation(rfclf_hp, "Random Forest (Tuned)")

# Support Vector Machine
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['rbf'],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
}

tune_model(svmclf, param_grid, "Support Vector Machine")
svmclf_hp = SVC(C=100, kernel="rbf", gamma=1)
svmclf_hp.fit(X_train_tr, y_train)
cross_validation(svmclf_hp, "Support Vector Machine (Tuned)")

model = max(score_list, key=lambda x:x['score'])
print(f"Most accurate model: { model.get('name') } with score " + "{:.2%}".format(model.get('score')))