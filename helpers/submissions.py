from ast import Raise
from email import errors
from enum import unique
from operator import is_
import os
import random
from narwhals import col
import pandas as pd
from datetime import datetime
from helpers.utils import sklearn_helper
from lightgbm import LGBMClassifier

import numpy as np


def prepare_submission(model, X_test, X_id, name, folder="submissions") -> str:

    expected_f, given_f = set(list(X_test.columns)), set(list(model.feature_name_))
    if expected_f != given_f:
        error_msg = (
            f"Model features do not match test data columns.\n"
            f"Unexpected features: {list(given_f - expected_f)}\n"
            f"Missing features: {list(expected_f - given_f)}"
        )
        raise ValueError(error_msg)

    predictions = model.predict_proba(X_test)[:, 1]

    submission_df = pd.DataFrame({"SK_ID_CURR": X_id, "TARGET": predictions})
    date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_path = os.path.join(folder, f"sub_{date}_{name}.csv")

    submission_df.to_csv(file_path, index=False)
    print(f"Submission created: {file_path}")
    return file_path


def lgbm_submission(X_train, y_train, X_test, name) -> str:
    """Trains a LightGBM model and prepares a submission file.
    handles categoricals, target and indexing columns"""
    X_train, y_train, X_test = X_train.copy(), y_train.copy(), X_test.copy()

    for col in X_train.select_dtypes(include=["category"]).columns:
        unique_cat = set(X_train[col].unique()) | set(X_test[col].unique())
        if np.nan in unique_cat:
            unique_cat.remove(np.nan)
        unique_cat = list(unique_cat)

        X_train[col] = X_train[col].astype("category").cat.set_categories(unique_cat)
        X_test[col] = X_test[col].astype("category").cat.set_categories(unique_cat)


    id_test = X_test.pop("SK_ID_CURR")
    X_test = X_test.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    X_train = X_train.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

    X_train = X_train[X_train.columns]
    X_test = X_test[X_train.columns]

    model = LGBMClassifier(is_unbalanced=True, random_state=3, verbose=-1)
    model.fit(X_train, y_train)

    return prepare_submission(model, X_test, id_test, name)


def cv_lgbm(X_train, y_train=None) -> pd.DataFrame:
    """Performs cross-validation with LightGBM and returns the score."""
    X_train = X_train.copy()
    if "TARGET" in X_train.columns:
        y_train = X_train.pop("TARGET")

    assert y_train is not None, "y_train must be provided if 'TARGET' is not in X_train"

    return sklearn_helper.stratified_cv_model(
        LGBMClassifier(is_unbalance=True, random_state=3, verbose=-1),
        X_train,
        y_train,
        scoring=["average_precision", "roc_auc", "f1_macro"],
    )
