import pandas as pd
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from dotenv import load_dotenv
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
WHO_PATH = "who.csv"

load_dotenv()

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)


should_save = os.environ.get('SAVE_FIGS', "true").lower() == "true"


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    if should_save:
        path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
        plt.clf()
        return
    print(f"Skipping {fig_id}... Saving is disabled.")


def load_data(data_path=WHO_PATH):
    data = pd.read_csv(data_path)
    return data.drop("id", axis=1)


who_data = load_data()

RANDOM_STATE = int(os.environ.get('RANDOM_STATE', '48'))


who_data.hist(bins=50, figsize=(20, 15))
save_fig("base_histogram_plots")


# Let's encode our text features and scale our numeric features:

numeric_features = ["age", "avg_glucose_level", "bmi"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")),
           ("scaler", StandardScaler())]
)
transformer = make_column_transformer(
    (numeric_transformer, numeric_features),
    (OneHotEncoder(drop='if_binary'), [
     'gender', 'work_type', 'Residence_type', 'smoking_status']),
    (OrdinalEncoder(), ['ever_married']),
    remainder='passthrough')

processed_who_data = transformer.fit_transform(who_data)

processed_who_data = pd.DataFrame(
    processed_who_data, columns=transformer.get_feature_names_out().tolist())


corr_matrix = processed_who_data.corr()
# most correlated features
print("CORRELATIONS\n", corr_matrix["remainder__stroke"].sort_values(
    ascending=False))
# The most correlated feature is AGE


processed_who_data["age_cat"] = pd.cut(
    processed_who_data["pipeline__age"], 4, labels=[1, 2, 3, 4])

processed_who_data.hist(bins=100, figsize=(40, 30))
save_fig("processed_histogram_plots")

processed_who_data["age_cat"].hist(bins=10, figsize=(20, 15))
save_fig("stratified age")

X = processed_who_data.drop(["remainder__stroke"], axis=1)
y = processed_who_data["remainder__stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=processed_who_data["age_cat"])

for Xi in (X_train, X_test, X):
    Xi.drop("age_cat", axis=1, inplace=True)


score_list = list()


def add_score(acc, title):
    score_list.append({"score": acc, "name": title})


def cross_validation(model, title):
    scores = cross_val_score(model, X, y, cv=5)
    acc = scores.mean()
    print("%s accuracy: %0.4f (+/- %0.4f)" % (title, acc, scores.std() * 2))
    add_score(acc, title)
    return acc


    # K-Nearest
knc_clf = KNeighborsClassifier()
cross_validation(knc_clf, "K-nearest neighbors")

# Logistic regression
lr_clf = LogisticRegression()
cross_validation(lr_clf, "Logistic regression")

# Decision Tree
dtc_clf = DecisionTreeClassifier()
cross_validation(dtc_clf, "Decision Tree")

# Random Forest
rfc_clf = RandomForestClassifier()
cross_validation(rfc_clf, "Random Forest")

# SVM
svc_clf = SVC()
cross_validation(svc_clf, "SVC")


print("PRELIMINARY RESULTS")
score_list.sort(key=lambda score: score["score"], reverse=True)
for i, result in enumerate(score_list):
    print("%s. %s: %0.4f" % (i+1, result["name"], result["score"]))

# LET's TUNE Log reg, SVC, and Rand For
# First we do random to have an idea where to test, then grid search

n_iter = 50
final_score_list = list()


def tune_model(model, params, title):
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid_search.fit(X_train, y_train)
    acc = grid_search.best_score_
    print(
        f"\n{title}:\n\tBEST PARAMS: {grid_search.best_params_}\n\tTRAIN SCORE:{acc}\n\t")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, labels=np.unique(y_pred))
    pre = precision_score(
        y_test, y_pred, average='weighted', labels=np.unique(y_pred))

    conf_mat = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        conf_mat, display_labels=best_model.classes_)
    display.plot()
    save_fig(f"confusion_matrix_{title}")

    final_score_list.append({"accuracy": acc, "recall": rec, "precision": pre,
                             "average": np.mean([acc, rec, pre]),  "name": title})
    print("\n\tTEST DATA RESULTS:\n\tACCURACY: %0.6f \t RECALL: %0.4f \t PRECISION: %0.6f" %
          (acc, rec, pre))


# Logistic Regression
lr_params = {
    "C": list(np.arange(0.1, 1, 0.01)),
    "class_weight": [None],
    "solver": ["lbfgs"],
}

lr_rscv = RandomizedSearchCV(
    estimator=lr_clf, param_distributions=lr_params, n_iter=n_iter, random_state=RANDOM_STATE)


lr_rscv.fit(X_train, y_train)

lr_params["C"] = [0.1, 0.2, 0.3, 0.4, 0.5]

tune_model(lr_clf, lr_params, "Logistic Regression")

# SVC
svc_params = {
    "C": list(np.arange(1, 100, 1)),
    "kernel": ["poly", "rbf"],
    "degree": [3, 5, 7, 9, 10],
    "gamma": ["scale", "auto"],
}


svc_rscv = RandomizedSearchCV(
    estimator=svc_clf, param_distributions=svc_params, n_iter=n_iter, random_state=RANDOM_STATE)


svc_rscv.fit(X_train, y_train)

print("SVC RANDOM SEARCH: ", svc_rscv.best_params_)

svc_params = {
    "C": [0.5, 1, 1.5],
    "kernel": ["rbf"],
    "gamma": ["scale"]
}

tune_model(svc_clf, svc_params, "SVC")

# Random Forest
rfc_params = {
    "n_estimators": list(range(1, 1000, 10)),
    "max_features": ["sqrt", "log2"]
}

""" #(commented to execute script faster)
rfc_rscv = RandomizedSearchCV(
estimator=rfc_clf, param_distributions=rfc_params, n_iter=n_iter, random_state=RANDOM_STATE, n_jobs=2)

rfc_rscv.fit(X_train, y_train)

print("RANDOM FOREST RANDOM SEARCH: ", rfc_rscv.best_params_)
"""
rfc_params = {
    "n_estimators": [50, 75, 100, 125],
    "max_features": ["sqrt", "log2"]
}

tune_model(rfc_clf, rfc_params, "Random Forest")
