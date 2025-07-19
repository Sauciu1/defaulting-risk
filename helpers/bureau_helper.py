import pandas as pd


def encode_agg_bureau_balance(
    data: pd.DataFrame,
    lookback: str = "10",
    forecast: str = "3",
    include_end_month: bool = False,
    X_y_split: bool = False,
    test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
    Create sequences for time series prediction using vectorized operations.
    lookback: number of months to look back for features
    forecast: number of months to predict forward
    """

    df = data.sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"]).reset_index(drop=True)


    # Add target columns for forecasting
    for i in range(forecast):
        df[f"t_{i}"] = df.groupby("SK_ID_BUREAU")["STATUS"].shift(-i)

    # Include first forecasted month in the features
    if include_end_month:
        df["first_pred_month"] = df["MONTHS_BALANCE"]

    # Create features for the past 'lookback' months
    for i in range(lookback):
        df[f"f_{i}"] = df.groupby("SK_ID_BUREAU")["STATUS"].shift(-i + lookback)

    df.drop(columns="STATUS", inplace=True)
    df = df[sorted(df.columns)]

    # Remove rows with NaN in target columns
    target_cols = [f"t_{i}" for i in range(forecast)]
    df = df.dropna(subset=target_cols)


    if X_y_split:
        if test:
            df = df[df["MONTHS_BALANCE"] == -2]

        X = df.drop(
            columns=target_cols + ["SK_ID_BUREAU", "MONTHS_BALANCE"], errors="ignore"
        )
        y = df[target_cols]

        return X, y

    return df
