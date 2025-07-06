import seaborn as sns
import matplotlib.pyplot as plt

def plot_categorical_dist(data, feature, hue=None, title = None):
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
    return data[feature].value_counts()



def plot_numeric_dist(data, feature, hue = None, title=None):
    plt.figure(figsize=(8, 3))

    ax = sns.histplot(data = data, hue = hue, x= feature, kde=True, bins=30, )

    if title:
        plt.title(title)

    if hue:
        sns.move_legend(ax ,loc='center left', bbox_to_anchor=(1, 0.5))

    if abs(data[feature].max()-data[feature].min()) > 1000:
        ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout()
    return ax


from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import fisher_exact
import seaborn as sns
from scipy.stats import chi2_contingency