import pandas as pd
from pandas.plotting import scatter_matrix
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
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

processed_who_data.hist(bins=50, figsize=(40, 30))
save_fig("processed_histogram_plots")

processed_who_data["age_cat"].hist()
save_fig("stratified age")

X = processed_who_data.drop(["remainder__stroke"], axis=1)
y = processed_who_data["remainder__stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=processed_who_data["age_cat"])

for Xi in (X_train, X_test, X):
    Xi.drop("age_cat", axis=1, inplace=True)


""" Gotta look into binary classification algorithms:
https://machinelearningmastery.com/types-of-classification-in-machine-learning/#:~:text=The%20class%20for%20the%20normal,probability%20distribution%20for%20each%20example.

Popular algorithms are:
- Logistic Regression
- k-Nearest Neighbors
- Decision Trees
- Support Vector Machine
- Naive Bayes
"""

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

# Log reg:
lr_params = {
    "C": list(np.arange(0.1, 1, 0.01)),
    "class_weight": [None],
    "solver": ["lbfgs"]
}
n_iter = 50
score_list = list()
def tune_model(model, params, title):
    grid_search = GridSearchCV(estimator=model, param_grid = params, cv=5)
    grid_search.fit(X_train, y_train)
    acc = grid_search.best_score_
    print(f"{title}:\n\tBEST PARAMS: {grid_search.best_params_}\n\tTRAIN SCORE:{acc}\n")    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # TODO: recall is not working properly (shows zero)
    rec = recall_score(y_test, y_pred, labels=np.unique(y_pred))
    pre = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    plot_confusion_matrix(best_model, X_test, y_test, cmap='GnBu')
    print(acc,rec,pre)
    save_fig(f"confusion_matrix_{title}")
    score_list.append({"accuracy": acc, "recall":rec, "precision":pre, "average":np.mean([acc, rec, pre]),  "name": title})
    print("\tTEST:\n\t ACCURACY: %0.4f \t RECALL: %0.4f \t PRECISION: %0.4f" % (acc,rec,pre))
    score_list.append({"score": acc, "name": title})


lr_rscv = RandomizedSearchCV(
    estimator=lr_clf, param_distributions=lr_params, n_iter=n_iter, random_state=RANDOM_STATE)


lr_rscv.fit(X_train, y_train)
# TEST 

lr_params["C"] = [0.1, 0.4, 0.6, 1.]

tune_model(lr_clf, lr_params, "Linear Regression" )