import pandas as pd
from scipy.stats import mannwhitneyu
from IPython.display import display
import numpy as np

def perform_mann_whitney(data, feature, hue = "Transported", alpha = 0.01):
    """Perform Mann-Whitney U test for a given feature."""
    if feature not in data.columns:
        raise ValueError(f"Feature '{feature}' not found in the DataFrame.")
    
    group1 = data.loc[data[hue] == True, feature].dropna()
    group2 = data.loc[data[hue] == False, feature].dropna()

    stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
    result_df = pd.DataFrame({
        "Feature": [feature],
        "Mann-Whitney U Statistic": [stat],
        "p-value": [p_value],
        "alpha": [alpha],
        "p < alpha": [p_value < alpha],
        "Group1 (Transported=True) median": [group1.median()],
        "Group2 (Transported=False) median": [group2.median()],
    }).T
    result_df.columns = ["Value"]
    result_df.index.name = "Metric"

    display(result_df)

    return stat, p_value;