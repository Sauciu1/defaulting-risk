import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_categorical_dist(data, feature, hue=None, title = None, show = True):
    plt.figure(figsize=(8, 1 + 0.5* len(data[feature].unique())))

    ax = sns.countplot(data=data, y=feature, hue=hue)
    if title:
        ax.set_title(title)
    total = len(data[feature])
    # Add percentages

    for c in ax.containers:
        labels = [f'{(v/total)*100:.1f}%' for v in c.datavalues]
        ax.bar_label(c, labels=labels)

    plt.title(feature, fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.ylabel(None)

    if hue and (hue!=feature):
        plt.legend(title=hue, loc='center left', bbox_to_anchor=(1.07, 0.5))
    elif hue==feature:
        plt.legend().remove()

    plt.tight_layout()
    if show:
        plt.show()
    return data[feature].value_counts()


def get_outlier_bounds(series, factor=1.5):
    q1 = series.quantile(0.05)
    q3 = series.quantile(0.95)
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr

    if low == high:
        print("!Warning: low and high bounds are equal. This may indicate a constant or nearly constant series.")
        return series.min(), series.max()

    if (low > series.min() )or (high < series.max()):
        print("!Servere outliers detected. Total values:", len(series), "original range:", series.min(), "-", series.max())
    if (series < low).any():
        print(f"Hiding outliers bellow {low}: {series[series < low].count()} values;")
    if (series > high).any():
        print(f"Hiding outliers above {high}: {series[series > high].count()} values;")


    low = max(low, series.min())
    high = min(high, series.max())

    return low, high

def plot_numeric_dist(data, feature, hue=None, title=None, show=True):


    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.copy()
    plt.figure(figsize=(8, 3))
    


    low, high = get_outlier_bounds(data[feature])
    data = data[(data[feature] >= low) & (data[feature] <= high)]

    discrete = len(data[feature].unique())<30
  
    ax = sns.histplot(data=data, hue=hue, x=feature, kde=True, discrete=discrete, bins=50)

    if title:
        plt.title(title)

    if hue:
        sns.move_legend(ax ,loc='center left', bbox_to_anchor=(1, 0.5))

    if abs(data[feature].max()-data[feature].min()) > 1000:
        ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout()
    if show:
        plt.show()

    return ax


from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import fisher_exact
import seaborn as sns
from scipy.stats import chi2_contingency