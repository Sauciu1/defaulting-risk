import optuna
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import joblib
import os
from sklearn.pipeline import Pipeline
from IPython.display import display

def logging_callback(study, frozen_trial, update_freq=20):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    call_count = study.user_attrs.get("logging_callback_count", 0) + 1
    study.set_user_attr("logging_callback_count", call_count)

    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
            )
        )

    elif call_count % update_freq == 0:
        print(f" Running Trial {call_count} with best value: {study.best_value}.")


def objective_function(
    trial: optuna.trial.Trial,
    model: BaseEstimator,
    params: callable,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str,
) -> float:
    
    model_clone = clone(model)
    model_clone[-1].set_params(**{f'classifier__{k}': v for k, v in params(trial).items()})
    # 5-fold stratified cross-validation
    train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train)
    model_clone.fit(train_x, train_y)
    preds = model_clone.predict(valid_x)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=326)
    scores = cross_val_score(model_clone, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    return np.mean(scores)

def copy_study_to_pipeline(study: optuna.study, pipeline:BaseEstimator):
    """
    Copies the model parameters from a fitted model to a pipeline's classifier step.
    """
    
    best_params = study.best_trial.params
    pipeline.named_steps['classifier'].set_params(**best_params)
    return pipeline


def load_or_create_end_model(pipe, model_name, creation_study: callable):
    model_path = os.path.join('models', model_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path):
        print("LGBM Model loaded from file: " + model_path)
        model = joblib.load(model_path)
    else:
        model = creation_study(pipe)
        print("LGBM Model saved to file: " + model_path)
        joblib.dump(model, model_path)

    display(model)
    return model


import pickle
def load_or_create_study(pipe, model_name, creation_study: callable):
    model_path = os.path.join('models', model_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path):
        print("LGBM Model loaded from file: " + model_path)
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    else:
        model = creation_study(pipe)
        print("LGBM Model saved to file: " + model_path)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

    display(model)
    return model