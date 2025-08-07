import pandas as pd


def encode_agg_bureau_balance(
    data: pd.DataFrame,
    lookback: int = 10,
    forecast: int = 3,
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




    # Create features for the past 'lookback' months
    for i in range(lookback):
        df[f"f_{lookback-i-1}"] = df.groupby("SK_ID_BUREAU")["STATUS"].shift(-i + lookback)


        # Add target columns for forecasting
    for i in range(forecast):
        df[f"t_{i}"] = df.groupby("SK_ID_BUREAU")["STATUS"].shift(-i)

    # Include first forecasted month in the features
    #if include_end_month:
    #    df["first_pred_month"] = df["MONTHS_BALANCE"]

    df.drop(columns="STATUS", inplace=True)

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


def simplify_credit_type(df) -> pd.DataFrame:
    """Agregates credit type"""
    df = df.copy()
    df["CREDIT_TYPE"] = df["CREDIT_TYPE"].astype("str")
    df.loc[~df["CREDIT_TYPE"].isin(["Consumer credit", "Credit card", "Mortgage"]), "CREDIT_TYPE"] = "Other"
    df["CREDIT_TYPE"] = df["CREDIT_TYPE"].astype("category")
    return df


def process_bureau_balance(balance_df) -> pd.DataFrame:
    """Adds bureau balance aggregated data to bureau dataframe"""


    balance_df = balance_df.loc[:, ["SK_ID_BUREAU","f_1", "f_0", "bad_payments"]]

    balance_df.columns = ["SK_ID_BUREAU"] + ["bur_bal_"+col for col in balance_df.columns[1:]]
    
    balance_df["SK_ID_BUREAU"] = balance_df["SK_ID_BUREAU"].astype(str)

    return balance_df