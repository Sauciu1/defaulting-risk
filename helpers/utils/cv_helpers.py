from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import os
from IPython.display import display
plt.style.use('fivethirtyeight')
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay

def get_metrics(y, y_pred):
    """Calculate accuracy, recall, precision, and F1 score for a given model and dataset."""
    acc = accuracy_score(y, y_pred)
    rec = recall_score(y, y_pred)
    prec = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    metrics = {
        'acc': acc,
        'recall': rec,
        'precision': prec,
        'f1': f1
    }
    return metrics



def cv_ROC_plot(model, X, y, n_cv=5, plot_curve =True):

    """Cross-validated ROC curve for a given model and dataset."""
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=3)
    roc_auc_scores = []
    pc_auc_scores = []

    if plot_curve:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))


    for train_idx, test_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx])

        y_proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        y_test= y.iloc[test_idx]

        if plot_curve:
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, alpha=0.8)
            PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax2, alpha=0.8)

        auc = roc_auc_score(y_test, y_proba)
        roc_auc_scores.append(auc)
        pc_auc = average_precision_score(y_test, y_proba)
        pc_auc_scores.append(pc_auc)


        
    if hasattr(model, "predict_proba"):
        mean_auc = np.mean(roc_auc_scores)
        print(f"Mean ROC AUC: {mean_auc:.4f} ± {np.std(roc_auc_scores):.4f}")
        print(f"Mean Precision-Recall AUC: {np.mean(pc_auc_scores):.4f} ± {np.std(pc_auc_scores):.4f}")

    if plot_curve:
        ax.plot([0, 1], [0, 1], linestyle="--", color="r", alpha=0.8, label="Chance")
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Cross-validated ROC")
        ax.legend(loc="lower right")

    return 



def print_metrics(model, X, y, n_cv=5):

    y_pred = cross_val_predict(model, X, y, cv=n_cv)

    if hasattr(model, "predict_proba"):
        y_proba = cross_val_predict(model, X, y, cv=n_cv, method="predict_proba")[:, 1]
    else:
        y_proba = cross_val_predict(model, X, y, cv=n_cv, method="decision_function")

    auc = roc_auc_score(y, y_proba)
    print(classification_report(y, y_pred))

def metrics_and_ROC(pipeline: Pipeline, X, y, n_cv=5, plot_curve=True):
    """Prints metrics and plots ROC curve for a given model and dataset."""
    X = X.copy()
    #print(X)
    # Automatically determine where the classifier model sits in the pipeline
    if isinstance(pipeline, Pipeline):
        # Find the index of the first step that is not a transformer (i.e., the classifier)
        classifier_idx = find_non_transformer_step_index(pipeline)
        if classifier_idx is not None and classifier_idx > 0:
            X = Pipeline(pipeline.steps[:classifier_idx]).fit_transform(X)
            pipeline = pipeline.steps[classifier_idx][1]
    pipeline.fit(X, y)
    print_metrics(pipeline, X, y, n_cv)
    cv_ROC_plot(pipeline, X, y, n_cv, plot_curve=plot_curve)


def find_non_transformer_step_index(pipeline: Pipeline):
    """Finds the index of the first step in the pipeline that is not a transformer."""
    if isinstance(pipeline, Pipeline):
        for idx, (name, step) in enumerate(pipeline.steps):
            if not hasattr(step, "fit_transform"):
                return idx
    raise ValueError("No non-transformer steps found in the pipeline.")


from scripts.utils import sklearn_helper

def model_curves(models, X, y, cv=5, figsize = (10, 5)):
    "Draws ROC and Precision-Recall curves for multiple models using cross-validation."

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize)
    for name, pipe in models.items():

        if not hasattr(pipe, "predict_proba"):
            print(f"Skipping {name} as it does not support predict_proba.")
            continue

        X_train = sklearn_helper._convert_types(X, pipe)

        y_proba_pred = cross_val_predict(pipe, X_train, y, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
        RocCurveDisplay.from_predictions(y, y_proba_pred, name=name, ax=ax)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        PrecisionRecallDisplay.from_predictions(y, y_proba_pred, name=name, ax=ax2)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")



    ax.set_title("Composite 5-Fold ROC Curves")
    ax2.set_title("Composite 5-Fold Precision-Recall Curves")


    plt.tight_layout()