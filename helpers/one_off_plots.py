from ast import Return
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_corr_table(df, suffix, ax) -> None:
    df = df.copy().select_dtypes(include=['number'])
    
    df = df[[col for col in df.columns if col.endswith(suffix)]]

    df.columns = [col.replace(suffix, '') for col in df.columns]
    
    corr = df.corr()
    sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=False, fmt=".2f", square=True, center =0, vmax=1, vmin=-1)

    #ax.set_yticklabels([])
    ax.set_xticklabels([])
    return corr





def plot_metric_comparison(df, metric_suffixes, sup_title, titles=None) -> None:
    fig, axes = plt.subplots(1, len(metric_suffixes), figsize=(16, 5))

    returns = {}
    
    for ax, suffix in zip(axes, metric_suffixes):
        returns[suffix] = plot_corr_table(df, suffix, ax)
        ax.set_title(suffix.replace('_', ' ').title())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    plt.suptitle(sup_title, fontsize=16, fontweight='bold')


    for ax in axes:
        ax.collections[0].colorbar.remove()

    for ax in axes[1:] :
        ax.set_yticklabels([])
        

    def add_colorbar(fig)-> None:
        cbar_ax = fig.add_axes([1.005, 0.1, 0.02, 0.8])  # Position: [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

    add_colorbar(fig)

    return returns