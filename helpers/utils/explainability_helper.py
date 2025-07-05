from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def compute_permutation_importances(model, X, y, feature_names=None, n_repeats=10, random_state=42, scoring=None):
    """
    Compute permutation importances for a fitted model.
    Works with AutoML models that handle preprocessing internally.
    Any parameters not used for prediction should be removed.
    """

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring
    )
    if feature_names is None:
        feature_names = X.columns if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
    importances_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return importances_df, result


def plot_feature_importances_df(importances_df, title = None, top_n=None, figsize=None):
    """Plots barplots of feature importance"""
    df = importances_df.copy()
    if top_n is not None:
        df = df.head(top_n)

    figsize = (8, max(6, 1+ 0.2 * len(df))) if figsize is None else figsize

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        y="feature",
        x="importance_mean",
        #xerr=1.96 * df["importance_std"],
        edgecolor="k"
    )

    ax.errorbar(
        df["importance_mean"],
        range(len(df)),
        xerr=1.96 * df["importance_std"],
        fmt='none',
        ecolor='black',
        capsize=6,
        elinewidth=2
    )

    title = "Permutation Feature Importances with 95% CI" if title is None else title
    ax.set_xlabel("Permutation Importance (mean)")
    ax.set_title()
    plt.tight_layout()
    plt.show()


def permutation_importance_boxplot(permutation_result, feature_names,metric =None, title=None, top_n=None, figsize=None):
    importances = permutation_result['importances']
    # importances shape: (n_features, n_repeats)
    n_features = importances.shape[0]
    features = np.array(feature_names)
    # Compute mean importances for sorting
    mean_importances = np.mean(importances, axis=1)
    sorted_idx = np.argsort(mean_importances)[::-1]
    if top_n is not None:
        sorted_idx = sorted_idx[:top_n]
    features = features[sorted_idx]
    importances = importances[sorted_idx]

    figsize = (10, max(6, 1 + 0.3 * len(features))) if figsize is None else figsize
    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=[importances[i] for i in range(len(features))],
        orient='h'
    )
    
    ax.set_yticklabels(features)
    ax.set_xlabel(f"Permutation Importance (decrease in {metric} score)")
    if title is None:
        title = "Permutation Feature Importances (Boxplot)"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
