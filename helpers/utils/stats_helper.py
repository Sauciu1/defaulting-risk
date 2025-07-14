import pandas as pd
from scipy.stats import mannwhitneyu
from IPython.display import display
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


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


def compare_ordering_metrics(df, metric_suffixes, test=stats.kendalltau):
    """
    Compute correlation metrics (Kendall or Spearman) for variable groups in a DataFrame.

    Parameters:
    - df: pandas.DataFrame, the input DataFrame containing the variables.
    - metric_suffixes: list of str, suffixes for the variable groups (e.g., ['_AVG', '_MEDI', '_MODE']).
    - test: callable, the statistical test to use (default: stats.kendalltau).

    Returns:
    - pandas.DataFrame, a DataFrame containing correlation metrics and p-values.
    """
    # Extract the variable groups
    variable_groups = [
        col.replace(metric_suffixes[0], "") for col in df.columns if col.endswith(metric_suffixes[0])
    ]

    # Compute correlation metrics for each variable
    results = {}
    for variable in variable_groups:
        metrics = [f"{variable}{suffix}" for suffix in metric_suffixes]
        subset = df[metrics].dropna()

        results[variable] = {
            f"{metric_suffixes[i]} vs {metric_suffixes[j]}": test(subset[metrics[i]], subset[metrics[j]])
            for i in range(len(metric_suffixes)) for j in range(i + 1, len(metric_suffixes))
        }
        results[variable]['p_value'] = sum(
            result.pvalue for result in results[variable].values()
        )


    

    # Create a DataFrame to display results
    correlation_df = pd.DataFrame.from_dict(
        {
            variable: {
                comparison: result.correlation
                for comparison, result in correlations.items() if comparison != 'p_value'
            }
            for variable, correlations in results.items()
        },
        orient="index",
    )

    # Add p-values to the DataFrame
    correlation_df['Bonferroni sum p_value'] = [
        correlations['p_value']*len(results.values()) for correlations in results.values()
    ]

    # Add a row called 'column mean' with the mean of each column
    correlation_df.loc['column mean', :] = correlation_df.mean(numeric_only=True)
    correlation_df.index.name = "Feature"

    return correlation_df


def plot_ordered_metrics(data, title, figsize=(10, 12)):
    
    data = data.reset_index().melt(id_vars='Feature', var_name='Comparison', value_name='Correlation')
    data = data[data.Comparison != 'Bonferroni sum p_value']

    plt.figure(figsize=figsize)

    sns.barplot(data=data, x='Correlation', y='Feature', hue='Comparison')
    
    plt.title(title)

    for container in plt.gca().containers:
        plt.bar_label(container, fmt='%.2f', label_type='edge')


    plt.legend(title='Metric Comparison', loc='upper left', bbox_to_anchor=(1.05, 1))