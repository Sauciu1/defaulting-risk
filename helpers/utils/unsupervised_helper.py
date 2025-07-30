from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .stats_helper import clip_IQR_outliers


def pca_3d(data) -> tuple[PCA, np.ndarray]:
    """Perfroms PCA for 3 components"""
    X = data.select_dtypes(include=["number"]).dropna()

    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)

    principal_components = pca.fit_transform(X)
    return pca, principal_components


def plotly_3d_pca(
    data: pd.DataFrame, hue_col: str = None, figsize: tuple[int, int] = (700, 550)
) -> go.Figure:
    """Plots 3 component PCA"""
    print("Plotting 3D PCA with discarded NANs")
    X = data.select_dtypes(include=["number"]).dropna()
    if hue_col is not None:
        hue_col = data.loc[X.index, hue_col]

    pca, principal_components = pca_3d(X)

    fig = px.scatter_3d(
        x=principal_components[:, 0],
        y=principal_components[:, 1],
        z=principal_components[:, 2],
        labels={
            "x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
            "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
            "z": f"PC3 ({pca.explained_variance_ratio_[2]:.1%} var)",
        },
        title="3D PCA of Bureau Credit Features",
        opacity=0.4,
        color=hue_col,
    )

    if hue_col is not None:
        fig.update_layout(legend_title_text=hue_col.name)

        # Add percentage of all values to each hue col name in legend
        value_counts = hue_col.value_counts(normalize=True)
        new_names = {
            str(val): f"{val} ({value_counts[val]:.1%})" for val in value_counts.index
        }
        fig.for_each_trace(
            lambda trace: trace.update(name=new_names.get(trace.name, trace.name))
        )

    fig.update_traces(marker=dict(size=2))
    fig.update_layout(width=figsize[0], height=figsize[1])
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=-0.3)),
        legend=dict(
            itemsizing="constant",
        ),
        # scene=dict(domain=dict(x=[0.1, 0.9], y=[0.1, 1])),
    )

    return fig



