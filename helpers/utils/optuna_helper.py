from multiprocessing import Pipe
import optuna
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import joblib
import os
from sklearn.pipeline import Pipeline
from IPython.display import display
import joblib
from sklearn.metrics import get_scorer

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
    cv: bool|int = False
) -> float:
    
    model_clone = clone(model)

    if isinstance(model_clone, Pipeline):
        model_clone[-1].set_params(**{f'classifier__{k}': v for k, v in params(trial).items()})
    else:
        model_clone.set_params(**params(trial))
    
    if (cv is False) or (cv is None) or (cv == 0):
        train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train)
        model_clone.fit(train_x, train_y)
        scorer = get_scorer(scoring)
        return scorer(model_clone, valid_x, valid_y)

    if cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=326)
        scores = cross_val_score(model_clone, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        return np.mean(scores)

def copy_study_to_pipeline(study: optuna.study, pipeline:BaseEstimator) -> BaseEstimator:
    """
    Copies the model parameters from a fitted model to a pipeline's classifier step.
    """
    
    best_params = study.best_trial.params
    pipeline.named_steps['classifier'].set_params(**best_params)
    return pipeline


def run_study(pipeline, params, X_train, y_train, scoring_function='roc_auc', n_trials = 50) -> dict:
    study = optuna.create_study(direction="maximize")
    
    def objective(trial) -> float:
        return objective_function(trial, pipeline, params, X_train, y_train, scoring_function)

    study.optimize(objective, n_trials=n_trials, )#callbacks=[logging_callback])

    return study



def load_or_create_study(name, run_study, path="models") ->Pipeline:
    full_path = f"{path}/{name}.jbl"
    if os.path.exists(full_path):
        with open(full_path, "rb") as f:
            return joblib.load(f)
    else:
        pipe = run_study()
        with open(full_path, "wb") as f:
            joblib.dump(pipe, f)
        return pipe
