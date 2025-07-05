from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import fisher_exact
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from matplotlib.axes import Axes


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


def fishers_test_vs_rest(contingency_table):
    """Perform Fisher's exact test for each category of a categorical feature against the rest."""
    results = dict()
    for idx, category in enumerate(contingency_table.index):
        count = contingency_table.loc[category, True]
        nobs = contingency_table.loc[category].sum()
        rest_count = contingency_table[True].sum() - count
        rest_nobs = contingency_table.sum().sum() - nobs

        table = np.array([[count, nobs - count], [rest_count, rest_nobs - rest_count]])
        stat, pval = fisher_exact(table)

        results[category] = {"statistic": stat, "p-value": pval}

    return results



def get_ci_from_contingency(contingency_table, hue: str, ci_multiplier=1.96):
    """Calculate confidence intervals for the ratio of True to False in a contingency table."""
    counts = contingency_table.loc[:, True]
    ns = contingency_table.sum(axis=1)
    ci_low, ci_upp = [], []

    for count, n in zip(counts, ns):
        # CI for binomial proportion
        prop = count / n
        se = np.sqrt(prop * (1 - prop) / n)
        ci_low.append(prop - ci_multiplier * se)
        ci_upp.append(prop + ci_multiplier * se)

    return ci_low, ci_upp


def plot_CI_ratio(ax: Axes, contingency_table, hue: str, ci_multiplier=1.96):
    """Plot confidence intervals for the ratio of True to False in a contingency table."""
    proportions = contingency_table.loc[:, True] / contingency_table.sum(axis=1)
    ci_low, ci_upp = get_ci_from_contingency(contingency_table, hue, ci_multiplier)

    # Convert boolean indices to string for x-axis if necessary
    x_labels = proportions.index.astype(str) if proportions.index.dtype == bool else proportions.index

    ax.errorbar(
        x_labels,
        proportions.values,
        yerr=[
            proportions.values - np.array(ci_low),
            np.array(ci_upp) - proportions.values,
        ],
        fmt="o",
        capsize=10,
        capthick=5,
        elinewidth=5,
        markersize=12,
    )
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    return ci_low, ci_upp

def max_ci_width(ci_low, ci_upp):
    """Calculate the maximum width of confidence intervals."""
    return max(np.array(ci_upp) - np.array(ci_low))

def set_y_axis_lim(ci_low, ci_upp, ax):
    """Set y-axis limits based on confidence intervals."""
    potential_upper = max(ci_upp) + max_ci_width(ci_low, ci_upp) * 0.35
    ymax = max(potential_upper, ax.get_ylim()[1])
    ymin = min(min(ci_low) - max_ci_width(ci_low, ci_upp)*0.2, ax.get_ylim()[0])
    ax.set_ylim(ymin, ymax)

def plot_overall_CI(data: pd.DataFrame, hue: str, ax: Axes):
    """Plot CI interval for hue only (overall proportion)."""
    contingency_table = pd.crosstab(index=[0], columns=data[hue])
    plot_CI_ratio(ax, contingency_table, hue=hue)
    ax.set_title(f"Overall")
    ax.set_xlabel("All Data")
    ax.set_xticks([0])
    ax.set_xticklabels(["All"])



def plot_feature_fishers(contingency_table: pd.DataFrame, feature: str, hue: str, ax: Axes):
    """Plot Fisher's test and CI for a single feature."""
    
    ax.set_xlim(-0.4, len(contingency_table) - 0.6)
    ax.set_title(f"by {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel(None)


    ci_low, ci_upp = get_ci_from_contingency(contingency_table, hue)

    # Fisher's exact test for each category vs rest
    fisher_results = fishers_test_vs_rest(contingency_table)
    ns = contingency_table.sum(axis=1)


    for idx, (category, result) in enumerate(fisher_results.items()):
        stat, pval = result["statistic"], result["p-value"]
        # Bonferroni correction for multiple comparisons
        pval *= len(fisher_results)
        pval = min(pval, 1.0)
        # Display p-value above the CI bar
        ax.text(
            idx,
            ci_upp[idx] + max_ci_width(ci_low, ci_upp) * 0.02,
            f"p={pval:.4f}",
            ha="center",
            va="bottom",

            fontsize=12,
            color="red" if pval < 0.05 else "black",
 
        )


        ax.text(
            idx,
            ci_low[idx] - max_ci_width(ci_low, ci_upp) * 0.1,
            f"n={ns[category]}",
            ha="center",
            va="top",
            fontsize=11,
            color="black",
       
        )
        subtest_result = pd.DataFrame(
            {"feature": [f"{feature}:{category}"], "chi2": [stat], "p-value": [pval]}
        )
        ax.grid(axis='x', visible=False)




    return ax, subtest_result


def plot_hue_basis(ax, data: pd.DataFrame, hue: str):
    """Plot the overall hue ratio and CI on given axis"""
    contingency_table = pd.crosstab(index=[0], columns=data[hue])
    
    contingency_table.index = ["Overall"]
   # print(contingency_table)
    plot_CI_ratio(ax, contingency_table, hue=hue)

    ax.set_title(f"Overall")
    ax.set_xlabel("All Data")
    ax.set_xticks([0])
    ax.set_xticklabels(["All"])
    ax.grid(axis='x', visible=False)
   


def plot_composite_features(data: pd.DataFrame, features: list[str], hue, fig_height=5, share_y=True):
    unique_cat = [1.5] + [len(data[cat].unique()) for cat in features]
    fig, axes = plt.subplots(
        1,
        len(features) + 1,
        figsize=(4 * sum(np.log(unique_cat))*0.8, fig_height),
        sharey=share_y,
        width_ratios=np.log(unique_cat),
    )

    # Plot overall CI
    plot_hue_basis(axes[0], data, hue)
    
    for feature, ax in zip(features, axes[1:]):
        if feature != "stroke":
            contingency_table = pd.crosstab(data[feature], data[hue])
            plot_feature_fishers(contingency_table, feature, hue=hue, ax=ax)

            plot_CI_ratio(ax, contingency_table, hue)

            ci_low, ci_upp = get_ci_from_contingency(contingency_table, hue)
            set_y_axis_lim(ci_low, ci_upp, ax)

    plt.suptitle(
        f"Relative {hue.title()} Frequency by Categorical Features and 95% CI\n"
        "(p-values from Fischer's Exact Test after Bonferroni Correction)",
        fontsize=15,
        y=1.05
    )
    axes[0].set_ylabel(f"Proportion of Stroke patients".title())

    

def rotatate_xticks(angle=45):
    """Rotate x-ticks for all axes in the provided list."""
    for ax in plt.gcf().axes:
        for label in ax.get_xticklabels():
            label.set_rotation(angle)