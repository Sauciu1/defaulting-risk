import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
import shap
import numpy as np
import pandas as pd

def feature_importance(model, X, preprocessor = None):
    """Plots feature importance using SHAP values for a given model and dataset."""

    if model is lgb.sklearn.LGBMClassifier:
        model = model
    elif isinstance(model, Pipeline):
        model = model.named_steps['classifier']
    explainer = shap.TreeExplainer(model)


    
    

    if preprocessor:
        X = preprocessor.transform(X)
        feature_names = preprocessor.named_steps['categorical'].get_feature_names_out()
        
    else:
        feature_names = X.columns

    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar")

    return shapley_feature_ranking(shap_values, X)



def shapley_feature_ranking(shap_values, X):
    feature_order = np.argsort(np.mean(np.abs(shap_values), axis=0))
    return pd.DataFrame(
        {
            "features": [X.columns[i] for i in feature_order][::-1],
            "importance": [
                np.mean(np.abs(shap_values), axis=0)[i] for i in feature_order
            ][::-1],
        }
    )


def plot_feature_importance(pipeline:Pipeline, X, y):
    X= X.copy()
    assert isinstance(pipeline, Pipeline), "pipeline must be a sklearn Pipeline"


    #model = pipeline.named_steps['classifier']
    # Apply all steps before 'classifier' to X to get observations
    pipeline.fit(X, y)

    for name, step in list(pipeline.named_steps.items()):
        if name == 'classifier':
            break
        X= step.fit_transform(X)

    
    print(pipeline[-2].get_feature_names_out())

    pipeline[-1].feature_name = pipeline[-2].get_feature_names_out()

  
    lgb.plot_importance(pipeline[-1], importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance (Gain)")
  


