from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from IPython.display import Image, display
from matplotlib.ticker import MaxNLocator


def set_plot_style():
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 15
    plt.style.use("fivethirtyeight")


def quantile_pair_grid(
    data: pd.DataFrame,
    columns: list[str],
    hue: str = None,
    lower_quantile: list[float] = None,
    upper_quantile: list[float] = None,
    figsize=(13, 12),
    title: str = None,
):
    fig = plt.figure(figsize=figsize)

    """
    Generate a pairgrid for the specified columns in the data.
    The x and y limits are set to the upper and lower quantiles.
    """
    assert (
        len(columns) == len(lower_quantile) == len(upper_quantile)
    ), "Length of columns, lower_quantile, and upper_quantile must be the same"

    data = data.copy()
    for i, col in enumerate(columns):
        data[col] = data[col].clip(
            lower=data[col].quantile(lower_quantile[i]),
            upper=data[col].quantile(upper_quantile[i]),
        )

    g = sns.PairGrid(data, vars=columns, diag_sharey=False, hue=hue)
    g.map_lower(sns.scatterplot, alpha=0.5)
    g.map_diag(sns.histplot, kde=True, bins=30, alpha=0.5)
    g.map_upper(sns.kdeplot, alpha=0.5)

    g.add_legend(
        title=hue, adjust_subtitles=True, bbox_to_anchor=(1.05, 0.5), loc="center left"
    )

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    return fig


def show_or_generate_image(image_path, generate_func, *args, **kwargs):
    """
    If the image at image_path exists, display it.
    Otherwise, run generate_func(*args, **kwargs) to create and display the image.
    """
    if os.path.exists(image_path):
        return Image(filename=image_path)
    else:
        fig = generate_func(*args, **kwargs)
        plt.savefig(image_path)

        return fig


def normalize_row_ylim(ax):
    """Normalizes the y limits for each row"""
    for i in range(ax.shape[0]):
        y_lim = max([ax[i, j].get_ylim()[1] for j in range(ax.shape[1])])
        for j in range(ax.shape[1]):
            ax[i, j].set_ylim(0, y_lim)


def double_y_ticks(ax):
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            # Get current number of ticks and double it
            current_ticks = (
                ax[i, j].yaxis.get_major_locator().nbins
                if hasattr(ax[i, j].yaxis.get_major_locator(), "nbins")
                else 5
            )
            ax[i, j].yaxis.set_major_locator(
                MaxNLocator(
                    nbins=current_ticks, integer=True, prune=None, steps=[1, 2, 5, 10]
                )
            )


def per_column_countplot(
    data, columns, feature, hue, figsize=(14, 14), normalize_rows=True
):
    """
    Generate a count plot for each column in 'columns' against the specified 'feature'.
    """
    feat_cols = data[feature].dropna().unique()

    fig, ax = plt.subplots(len(columns), len(feat_cols), figsize=figsize, sharey=False)
    plt.subplots_adjust(wspace=0.10, hspace=0.10)

    for i, cat1 in enumerate(columns):
        for j, cat2 in enumerate(feat_cols):

            subset = data[data[feature] == cat2]
            sns.countplot(data=subset, x=cat1, hue=hue, ax=ax[i, j])

            ax[i, j].set_title(None)
            ax[i, j].legend_.remove()
            ax[i, j].set_xlabel(None)

            # Rotate x-tick labels if any label is longer than 10 characters
            xticklabels = [tick.get_text() for tick in ax[i, j].get_xticklabels()]
            if any(len(label) > 10 for label in xticklabels):
                ax[i, j].tick_params(axis="x", rotation=10)

    if normalize_rows:
        normalize_row_ylim(ax)
    double_y_ticks(ax)

    for i, col in enumerate(columns):
        ax[i, 0].set_ylabel(col, fontweight="bold")

    # if any(len(col) > 10 for col in columns):
    #    for i, col in enumerate(columns):
    #       ax[i, 0].set_ylabel(col, fontweight='bold', rotation=80)

    # ax[-1, i].set_xlabel(feature)
    for i, col in enumerate(feat_cols):
        ax[-1, i].set_xlabel(f"{feature} = {col}", fontweight="bold")

    plt.suptitle(
        f"Count plot of {feature} vs [{', '.join(columns)}]",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Add a single legend to the right of all subplots
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title=hue, loc="center left", bbox_to_anchor=(1.00, 0.7)
    )
    return fig


from typing import Literal





def plot_corr_triangle(
    data,
    method: Literal["pearson", "spearman"] = "pearson",
    response_var=None,
    figsize = None,
    heatmap_kws={"annot":False, "fmt": ".2f", "square": True, "ax":None},
)    -> None:
    """plots a triangular correlation matrix"""
    if response_var is not None:
        data = data[
            [col for col in data.columns if col != response_var] + [response_var]
        ]

    corr = data.corr(numeric_only=True, method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    if heatmap_kws.get("ax", None):
        ax = heatmap_kws["ax"]
        ax.grid(False)
        figsize = None
    elif figsize:
        plt.figure(figsize=figsize)



    sns.heatmap(
        corr,
        cmap="coolwarm",
        mask=mask,
        center=0,
        vmax=1,
        vmin=-1,
        **heatmap_kws
    )
    plt.grid(False)

    # Center the colorbar if present
    ax = plt.gca()
    cbar = ax.collections[0].colorbar if ax.collections else None
    if cbar:
        cbar.ax.set_position([0.78, 0.1, 0.03, 0.7])  # adjust as needed to center
        cbar.ax.set_title("Correlation", fontsize=12, fontweight="bold")
    
    plt.title(f"{method} Correlation Heatmap".title())


    #plt.grid(False)

    return corr


def count_lineplot(data, x, hue, ax=None, title=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        ax.set_title(title)

    # Group and count
    counts = data.groupby([x, hue]).size().reset_index(name="count")

    # Pivot for plotting
    pivot_df = counts.pivot(index=x, columns=hue, values="count").fillna(0).sort_index()

    # Plot each hue line separately
    for col in pivot_df.columns:
        sns.lineplot(
            data=pivot_df,
            x=pivot_df.index,
            y=col,
            ax=ax,
            label=str(col),
            marker="X",  # fixed marker argument
            markersize=13,  # fixed markersize argument
            **kwargs,
        )

    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    ax.legend(title=hue)
    return ax
