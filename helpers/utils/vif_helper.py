import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.patches import Patch


def calculate_vif(data, columns):
    """Calculates Variance Inflation Factor (VIF) for the given columns in a DataFrame."""

    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame"
    assert all(col in data.columns for col in columns), "missing columns in data"
    data = data[columns].copy().dropna(axis=0)
    assert all(data[col].dtype in [float, int] for col in columns), "All columns must be of type float or int"

    vif_data = pd.DataFrame()
    vif_data["Feature"] = columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)


def plot_vif(data, columns, figsize=(10, 6)):
    """Calculates and plots Variance Inflation Factor (VIF) for the given columns."""
    vif_data = calculate_vif(data, columns)

    plt.figure(figsize=figsize)
    sns.barplot(
        x="VIF",
        y="Feature",
        data=vif_data,
        #palette="viridis",
    )
    colors = vif_data["VIF"].apply(lambda x: "green" if x < 3 else ("yellow" if x < 10 else "red"))
    for bar, color in zip(plt.gca().patches, colors):
        bar.set_color(color)
    plt.xlabel("VIF")
    plt.title("Variance Inflation Factor (VIF) by Feature")
    plt.tight_layout()
    legend_elements = [
        Patch(facecolor='green', label='good'),
        Patch(facecolor='yellow', label='concerning'),
        Patch(facecolor='red', label='bad')
    ]
    plt.legend(handles=legend_elements, title="VIF Range")



    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)