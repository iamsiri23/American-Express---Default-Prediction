from catboost import Pool, CatBoostClassifier
from tqdm.notebook import tqdm_notebook
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from matplotlib.gridspec import GridSpec
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import log_loss
import optuna  # pip install optuna
from plotly.offline import init_notebook_mode
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors
import logging
import uuid
import joblib
import lightgbm as lgb
import catboost
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import warnings
from functools import partial, reduce
import glob
import pickle
import sys
import gc
import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest

# we use F-TEST for numerical variables
from sklearn.feature_selection import chi2, f_classif
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm
import random
from random import sample
import itertools
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask
import dask.dataframe as dd

train_data = dd.read_csv("train_data.csv")
train_data_df = train_data.compute()  # this converts dask to pandas

train_label = dd.read_csv("train_labels.csv")
train_label_df = train_label.compute()  # this converts dask to pandas
train_label_df = train_label_df.set_index(["customer_ID"])


def handling_missing_value(data):
    """
    Function: find number and percent of missing values in the data
    Input: data
    Output: variables with missing value and the missing percent
    """
    missing_number = data.isna().sum().sort_values(ascending=False)
    missing_percent = round(
        (data.isna().sum() / data.isna().count()).sort_values(ascending=False), 3
    )
    missing_values = pd.concat(
        [missing_number, missing_percent],
        axis=1,
        keys=["Missing_Number", "Missing_Percent"],
    )
    missing_df = missing_values[missing_values["Missing_Number"] > 0]
    return missing_df


def extract_date_cols(
    df, date_var="S_2", sort_by=["customer_ID", "S_2"], week_days={1: "Mon", 2: "Tue", 3: "Wen", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
):
    # change to datetime
    df[date_var] = pd.to_datetime(df[date_var])
    # sort by customer by date
    df = df.sort_values(by=sort_by)
    # extract some date characteristics
    # month
    df["month"] = df[date_var].dt.month
    # day of week
    df["day_of_week"] = df[date_var].apply(lambda x: x.isocalendar()[-1])
    return df


def generate_column_names_num(vars=num_vars, agg=["mean", "std", "min", "max", "last"]):
    tmp = []
    for a in agg:
        tmp_i = pd.Series(vars).apply(lambda x: x + "_" + a).tolist()
        tmp.append(tmp_i)
    column_names_num = list(itertools.chain(
        *zip(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4])))
    return column_names_num


# missing value count
missing_value_count = handling_missing_value(train_data_df)

# categorical variables
cat_vars = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
    "month",
    "day_of_week",
]

# feature columns
features = train_data_df.drop(["customer_ID", "S_2"], axis=1).columns.to_list()

# numerical variables
num_vars = list(filter(lambda x: x not in cat_vars, features))

# specific variables
delequincy_vars = filter(lambda x: x.startswith("D") and x not in cat_vars, features)
spend_vars = filter(lambda x: (x.startswith("S"))
                    and (x not in cat_vars), features)
payment_vars = filter(lambda x: x.startswith(
    "P") and x not in cat_vars, features)
balance_vars = filter(lambda x: x.startswith(
    "B") and x not in cat_vars, features)
risk_vars = filter(lambda x: x.startswith("R") and x not in cat_vars, features)

# extract date variables
train_data_df = extract_date_cols(
    train_data_df, date_var="S_2", sort_by=["customer_ID", "S_2"], week_days=week_days
)

train_data_df_cat = train_data_df[cat_vars]

# count of unique values for categorical variables
train_data_df[cat_vars].apply(lambda x: x.nunique(), axis=0)

# column names for numerical variables
column_names_num = generate_column_names_num(vars=num_vars)

# numerical variable aggregation
train_data_df_num = train_data_df.groupby("customer_ID", as_index=False)[
    num_vars].agg(["mean", "std", "min", "max", "last"])
train_data_df_num.columns = column_names_num


def select_features(X_train, y_train, X_test, k_value="all"):
    fs = SelectKBest(score_func=f_classif, k=k_value)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# feature selection
X_train_fs, X_test_fs, fs = select_features(X_dev_new, y_dev, X_test_new)
# what are scores for the features
for i in range(len(fs.scores_)):
    print("Feature %d: %f" % (i, fs.scores_[i]))


X_train_fs, X_test_fs, fs = select_features(X_dev_new, y_dev, X_test_new, 300)


features_selected = fs.get_feature_names_out()


X_dev_selected = X_dev_new[features_selected]


X_test_selected = X_test_new[features_selected]

def objective(trial, X, y):
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "binary_logloss")
            ],  # Add a pruning callback
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)


study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")


def func(trial):
    return objective(trial, X_dev_selected, y_dev)


study.optimize(func, n_trials=20)


print(f"\tBest value (rmse): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")


model = lgb.LGBMClassifier(objective="binary", **study.best_params)
model.fit(X_dev_selected, y_dev, eval_metric="binary_logloss")
preds = model.predict_proba(X_dev_selected)
log_loss(y_dev, preds)


pred_class_dev = model.predict(X_dev_selected)


print("***** Development Set Analysis *****")

test_acc = sum(np.where(pred_class_dev == y_dev, 1, 0)) / y_dev.shape[0]
print("Accuracy on Test Set : {:0.4f}".format(test_acc))
prfs = precision_recall_fscore_support(y_dev, pred_class_dev, average="macro")
print("Precision : ", prfs[0])
print("Recall : ", prfs[1])
print("F1 Score : ", prfs[2])


average_precision_score(y_dev, pred_class_dev_prob[:, 1])


pred_prob_dev = model.predict_proba(X_dev_selected)[:, 1]

fpr, tpr, _ = roc_curve(y_dev, pred_prob_dev)

print("AUC : ", auc(fpr, tpr))


pred_class_test = model.predict(X_test_selected)



print("***** Test Set Analysis *****")

test_acc = sum(np.where(pred_class_test == y_test, 1, 0)) / y_test.shape[0]
print("Accuracy on Test Set : {:0.4f}".format(test_acc))
prfs = precision_recall_fscore_support(y_test, pred_class_test, average="macro")
print("Precision : ", prfs[0])
print("Recall : ", prfs[1])
print("F1 Score : ", prfs[2])


pred_prob_test = model.predict_proba(X_test_selected)[:, 1]

fpr, tpr, _ = roc_curve(y_test, pred_prob_test)


pred_class_test = model.predict(X_test_selected)


pred_class_test_prob = model.predict_proba(X_test_selected)


lightgbm_test_prediction = pd.DataFrame(
    {
        "y_test": y_test,
        "y_pred_class": pred_class_test,
        "y_pred_prob": pred_class_test_prob[:, 1],
    }
)


lightgbm_test_prediction.to_csv("lightgbm_test_prediction.csv")

