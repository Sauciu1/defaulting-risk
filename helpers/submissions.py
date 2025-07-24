from ast import Raise
import os
from narwhals import col
import pandas as pd
from datetime import datetime
from helpers.utils import sklearn_helper
from lightgbm import LGBMClassifier


def prepare_submission(model, X_test, X_id, name, folder = 'submissions') ->str:


    expected_f, given_f = set(list(X_test.columns)), set(list(model.feature_name_))
    if expected_f != given_f:
        error_msg = (
            f"Model features do not match test data columns.\n"
            f"Unexpected features: {list(given_f - expected_f)}\n"
            f"Missing features: {list(expected_f - given_f)}"
        )
        raise ValueError(error_msg)


    predictions = model.predict_proba(X_test)[:, 1]


    submission_df = pd.DataFrame({'SK_ID_CURR': X_id, 'TARGET': predictions})
    date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    file_path = os.path.join(folder, f'sub_{date}_{name}.csv')

    submission_df.to_csv(file_path, index=False)
    print(f'Submission created: {file_path}')
    return file_path



def cv_lgbm(X_train, y_train=None)   -> pd.DataFrame:
    """Performs cross-validation with LightGBM and returns the score."""
    if 'TARGET' in X_train.columns:
        y_train = X_train.pop('TARGET')

    assert y_train is not None, "y_train must be provided if 'TARGET' is not in X_train"


    return sklearn_helper.stratified_cv_model(
    LGBMClassifier(is_unbalance=True, random_state=3, verbose=-1), 
    X_train, 
    y_train, 
    scoring=['average_precision', 'roc_auc', 'f1_macro']
)