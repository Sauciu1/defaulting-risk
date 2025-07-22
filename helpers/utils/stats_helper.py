import pandas as pd
from scipy.stats import mannwhitneyu
from IPython.display import display
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from typing import Literal


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


def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction to p-values.
    """
    corrected = p_values * len(p_values)
    return corrected.clip(upper=1)


def clip_IQR_outliers(X: pd.DataFrame, quantile: float = 0.25, multiplier: float = 1.5):
    """
    Clip outliers based on the IQR method.
    """
    Q1 = X.quantile(quantile)
    Q3 = X.quantile(1 - quantile)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return X.clip(lower=lower_bound, upper=upper_bound, axis=1)


def kruskal_wallis_cluster_separation(data, cluster_label_col, correction='Bonferroni'):
    """
    Performs Kruskal–Wallis H‑test for independent samples for each feature in the data,
    comparing the distribution of values across different clusters defined by cluster_label_col.
    """
    features = data.select_dtypes(include=[np.number]).columns
    results = pd.DataFrame(index=features)

    for feature in features:
        # Perform Kruskal-Wallis test for the feature across groups defined by cluster_label_col
        groups = [
            data[data[cluster_label_col] == label][feature].dropna()
            for label in data[cluster_label_col].unique()
        ]
        groups = [group for group in groups if len(group) > 0]  # Filter out empty groups
        try:
            _, p_value = stats.kruskal(*groups)
        except ValueError as e:
            if "All numbers are identical" in str(e):
                p_value = 1
            else:
                raise

        # Store the p-value for this feature (could use the first label as column, or create a summary column)
        results.loc[feature, 'p_value'] = p_value
        # Store the median for each group
        for idx, label in enumerate(data[cluster_label_col].unique()):
            if len(groups[idx]) > 0:
                results.loc[feature, f"{label}_median"] = groups[idx].median()
            else:
                results.loc[feature, f"{label}_median"] = np.nan

    results = results.astype(float)


    if correction == 'Bonferroni':
        results = bonferroni_correction(results)
    results = results.clip(upper=1)

    print('Corrected p-values for Wilcoxon test within clusters:')
    return results



def wilcoxon_test_within_cluster(data, features, cluster_label_col, target_col, correction = 'Bonferroni') -> pd.DataFrame:
    results = pd.DataFrame(index=features, columns=np.unique(data[cluster_label_col]))

    for feature in features:
        for label in data[cluster_label_col].unique():
            cluster_data = data[data[cluster_label_col] == label]
            groups = [
                cluster_data[cluster_data[target_col] == True][feature].dropna(),
                cluster_data[cluster_data[target_col] == False][feature].dropna()
            ]
            
            # Perform Wilcoxon rank-sum test between TARGET=True and TARGET=False within the cluster
            if len(groups[0]) > 0 and len(groups[1]) > 0:  # Ensure both groups have data
                _, p_value = stats.ranksums(groups[0], groups[1])
                results.loc[feature, label] = p_value

    results = results.astype(float)
    if correction == 'Bonferroni':
        results = bonferroni_correction(results)
    results = results.clip(upper=1)

    print('Corrected p-values for Wilcoxon test within clusters:')
    return results

import matplotlib.colors as mcolors
def heatmap_wilcoxon_cluster_results(wilcoxon_results, title="Wilcoxon Test Results Heatmap"):
    plt.figure(figsize=(12, 8))


    sns.heatmap(
        wilcoxon_results.astype(float),  # Ensure the data is numeric for heatmap
        annot=True,
        fmt=".1e",
        cbar_kws={"label": "p-value (log scale)"},
        vmax=0.05,
        norm=mcolors.LogNorm()
    )
    plt.title(title)
    plt.xlabel("Cluster Label")
    plt.ylabel("Feature")
    plt.xticks(rotation=45)
    plt.tight_layout()


def kruskal_wallis_test(data: pd.DataFrame, features: list, group_column: str, 
                       bonferroni_correction: bool = True) -> pd.DataFrame:
    """
    Perform Kruskal-Wallis test for multiple features across groups.
    """
    kruskal_results = {}
    
    for feature in features:
        groups = [
            group[feature].dropna()
            for name, group in data.groupby(group_column, observed=True)
        ]
        
        stat, p_value = stats.kruskal(*groups)
        
        # Apply Bonferroni correction if requested
        corrected_p_value = p_value * len(features) if bonferroni_correction else p_value
        
        kruskal_results[feature] = {
            "statistic": stat,
            "p_value": corrected_p_value,
            "raw_p_value": p_value if bonferroni_correction else None
        }
    
    return pd.DataFrame.from_dict(kruskal_results, orient="index")



def plot_kruskal_results(data: pd.DataFrame, cluster_label_col: str, correction: Literal['Bonferroni']='Bonferroni', figsize = (12, 8)):
    data = data.copy()
    kruskal_result = kruskal_wallis_cluster_separation(
        data=data,
        cluster_label_col=cluster_label_col,
        correction=correction
    )

    # Get numeric columns and scale them
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    #Clip ourliers for visualisation
    X_scaled = data[numeric_cols].copy()
    X_scaled = clip_IQR_outliers(data[numeric_cols], quantile=0.003, multiplier=2)

    X_scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(X_scaled), 
        columns=numeric_cols,
        index=data.index
    )


    # Add cluster labels to scaled data
    X_scaled[cluster_label_col] = data[cluster_label_col].values

    # Melt the dataframe for plotting
    scaled_df = pd.melt(
        X_scaled, 
        id_vars=[cluster_label_col], 
        value_vars=numeric_cols,
        var_name='feature', 
        value_name='value'
    )


    # Create the boxplot
    plt.figure(figsize=figsize)
    sns.boxplot(data=scaled_df, y='feature', hue=cluster_label_col, x='value', orient='h')
    plt.title('Feature Distribution by Cluster')
    plt.tight_layout()
    plt.xlabel('Scaled (0-1) Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')



    # Add p-value annotations to the plot
    for i, (feature, p_val) in enumerate(kruskal_result['p_value'].items()):
        if not pd.isna(p_val):
            plt.text(0.96, i, f'p={p_val:.3f}', transform=plt.gca().get_yaxis_transform(), 
                    ha='left', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    
    return kruskal_result