import pandas as pd

def neighbourhood_feature_selection(
    X: pd.DataFrame,
    labels: pd.Series,
    housing_columns: list[str],
    wilc_results: pd.DataFrame,
) -> pd.DataFrame:
    

    X["Cluster_Label"] = pd.Categorical(labels)

    for feature, row in wilc_results.iterrows():
        for cluster in row.index:
            if row[cluster] < 0.05:
                X[f"enc_cl{cluster}_{feature}"] = X[feature] * (
                    X["Cluster_Label"] == cluster
                ).astype(int)

    X.drop(columns=["Cluster_Label"] + housing_columns, inplace=True, errors="ignore")
    print(f"Selected {len(X.columns)} features after neighbourhood feature selection.")
    return X