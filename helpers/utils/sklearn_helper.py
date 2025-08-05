
from sklearn.model_selection import cross_validate
import pandas as pd
from typing import TypeVar, List, Dict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
Predictor = TypeVar('Predictor', bound='sklearn.base.BaseEstimator')
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from IPython.display import display



def _convert_types(X, model: Predictor) -> pd.DataFrame:
    """Accounts for SkLearn only working with object columns, while LightGBM works with categorical."""
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    assert hasattr(model, "__module__"), "model must have a __module__ attribute"
    X = X.copy()

    while model.__module__.startswith("sklearn.pipeline"):
        model = model[-1]
    
    if hasattr(model, "__module__") and "lightgbm" in model.__module__:
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category")

    else:
        for col in X.select_dtypes(include="category").columns:
            X[col] = X[col].astype("object")


    

    return X


def stratified_cv_model(model: Predictor, X: pd.DataFrame, y: pd.Series, scoring=['accuracy', 'roc_auc', 'f1'], cv: int = 5):
    probability_metrics = ['roc_auc']

    if not hasattr(model, "predict_proba"):
        scoring = list(set(scoring) - set(probability_metrics))

    X = _convert_types(X, model)
   

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_validate(
        model, X, y,
        scoring=scoring,
        cv=skf,
        return_train_score=False
    )
    mean = {metric: round(float(scores[f'test_{metric}'].mean()), 4) for metric in scoring}
    std = {metric: round(float(scores[f'test_{metric}'].std()), 4) for metric in scoring}
    results = pd.DataFrame([mean, std], index=["mean", "std"])
    return results





class ModelEvaluator:
    def __init__(self, eval_helper_func: callable):
        self.failed = []
        self.error_history = {}
        self.eval_func = eval_helper_func
        self.results = None

    def evaluate_models(self, models: dict) -> pd.DataFrame:
        results = []
        for i, (name, model) in enumerate(models.items()):
            try:

                print(f"Evaluating {name} - {i+1}/{len(models)}...")
                scores = self.eval_func(model)
                scores['model'] = name
                results.append(scores)
            except Exception as e:
                print(f"Error evaluating {name}. Skipping...")
                self.failed.append(name)
                self.error_history[name] = str(e)
                continue
        print(self.failed)
        self.results = pd.DataFrame(results).set_index('model').sort_values(by='accuracy', ascending=False)

        return self.results


from datetime import datetime
def prepare_csv_submission(model:Predictor, X_train:pd.DataFrame, y_train:pd.Series, X_test:pd.DataFrame, file_path_prefix = 'submission.csv')->pd.DataFrame:
    """Fits the model on training data and creates a submission CSV file."""

    X_train = _convert_types(X_train, model)
    X_test = _convert_types(X_test, model)

    model.fit(X_train, y_train)
    results = model.predict(X_test)
    
    submission = pd.DataFrame({
        'PassengerId': X_test['PassengerId'],
        'Transported': results
    })

    file_path = file_path_prefix.replace('.csv', '')+"-"+str(datetime.now().strftime('%Y-%m-%d_%H-%M'))+".csv"

    submission.to_csv(file_path, index=False)

    print("Submission file created at:", file_path)
    return submission


def make_pipelines(models:dict[str:Predictor], preprocessor):
    make_pipe = lambda model_cls: Pipeline(
        [("preprocessor", preprocessor), ("model", model_cls() if callable(model_cls) else model_cls)]
    )
    return {name: make_pipe(model_cls) for name, model_cls in models.items()}



def plot_top_models_scatter(model_scores, top_n=10, ax=None):
    """Scatterplot of accuracy vs f1 for top N models."""
    import seaborn as sns
    top = model_scores.sort_values(by="accuracy", ascending=False).head(top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        x=top["accuracy"],
        y=top["f1"],
        hue=top.index,
        style=top.index,
        s=100,
        ax=ax,
        alpha=0.7,
    )
    for model_name, row in top.iterrows():
        ax.text(
            row["accuracy"],
            row["f1"]+0.0013,
            model_name,
            fontsize=9,
            va="top",
            ha="center",
        )
    ax.get_legend().remove()
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("F1 Score")
    return top


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



def passthrough_transformer(categorical_vars: List[str], numeric_vars: List[str]):
    """Creates a ColumnTransformer that passes through specified categorical and numeric variables."""
    passthrough = ColumnTransformer(
        transformers=[
            ('categorical', 'passthrough', categorical_vars),
            ('numeric', 'passthrough', numeric_vars),
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    passthrough.set_output(transform="pandas")
    
    return passthrough
    


def make_passthrough_model(categorical_vars, numeric_vars, model):
    """Passes only the specified columns through without transformation."""

    return Pipeline([
        ('passthrough', passthrough_transformer(categorical_vars, numeric_vars)),
        ('classifier', model() if callable(model) else model)
    ])


from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import numpy as np

def make_impute_pipeline(categorical_vars, numeric_vars, model):
    """Creates a pipeline for imputing missing values in categorical and numeric variables."""

    
    
    pipe = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_vars,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_vars,
            ),
        ],
        remainder="drop",
    )
    pipe.set_output(transform="pandas")



    new_model = Pipeline(
        [
            ('passthrough', passthrough_transformer(categorical_vars, numeric_vars)),
            ("impute", pipe),
            ("classifier", model() if callable(model) else model),
        ]
    )

    return new_model


def one_hot_pipe(categorical_vars, numeric_vars, model):
    """Creates a preprocessing pipeline with one-hot encoding for categorical variables and scaling for numeric variables."""
    trans = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_vars,
            ),
            ("num", StandardScaler(), numeric_vars),
        ],
        remainder="drop",
    )

    return Pipeline(
        [   
            ("passthrough", passthrough_transformer(categorical_vars, numeric_vars)),
            ("preprocess", trans),
            ("classifier", model() if callable(model) else model),
        ]
    )

from sklearn.preprocessing import FunctionTransformer
from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def as_category(x):
    """Convert a Series to categorical type."""
    return x.astype("category")

def as_object(x):
    """Convert a Series to object type."""
    return x.astype("str")

def category_transformer(categorical_vars: List[str], astype:Literal["category", "str"] = "category"):
    """Creates a ColumnTransformer that applies a function to specified categorical variables."""
    
    converter_type = as_category if astype == "category" else as_object
    
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", FunctionTransformer(converter_type), categorical_vars)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")
    return transformer

