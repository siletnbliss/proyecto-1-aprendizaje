import pandas as pd
import sys
import os
import sklearn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
WHO_PATH = "who.csv"


# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def load_data(data_path=WHO_PATH):
    return pd.read_csv(data_path)


who_data = load_data()

RANDOM_STATE = int(os.environ.get('RANDOM_STATE', '48'))

who_data.hist(bins=50, figsize=(20, 15))
save_fig("base_histogram_plots")
# plt.show()

# stratify data: bmi, avg_glucose_level, age(?)

who_data["age_cat"] = pd.cut(who_data["age"], bins=[
                             0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

# replace N/A with mean value
who_data["bmi"] = who_data["bmi"].fillna((who_data["bmi"].mean()))

who_data["bmi_cat"] = pd.cut(who_data["bmi"], bins=[
                             0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

who_data["avg_glucose_level_cat"] = pd.cut(who_data["avg_glucose_level"], bins=[
    0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5]
)

plt.clf()

who_data["bmi_cat"].hist()
# save_fig("stratified_bmi")

plt.clf()

who_data["avg_glucose_level_cat"].hist()
# save_fig("stratified_glucose_level")

print(who_data["bmi_cat"].value_counts())
