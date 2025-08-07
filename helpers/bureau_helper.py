import pandas as pd

from helpers.preprocessing import load_pkl_to_preprocessor
import numpy as np

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



def credit_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = simplify_credit_active(df)
    df = choose_currency(df)

    
    df["credit_usage"] = (
        df["AMT_CREDIT_SUM"] / df["AMT_CREDIT_SUM_LIMIT"]
    ).replace(np.inf, np.nan)
    df["credit_duration"] = (
        df["DAYS_CREDIT"] - df["DAYS_ENDDATE_FACT"]
    )


    return df


def simplify_credit_active(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplify the CREDIT_ACTIVE column to only include 'Active', 'Closed', and 'Sold'.
    """
    df = df.copy()
    df["CREDIT_ACTIVE"] = (
        df["CREDIT_ACTIVE"]
        .astype("category")
        .cat.set_categories(["Active", "Closed", "Other"])
    )
    df.loc[~df["CREDIT_ACTIVE"].isin(["Active", "Closed"]), "CREDIT_ACTIVE"] = "Other"
    return df


def choose_currency(df: pd.DataFrame, currency: str = "currency 1") -> pd.DataFrame:
    """Choose only rows with the specified currency."""
    print(f"Filtering for currency: {currency}")

    return df[df["CREDIT_CURRENCY"] == currency]


def get_bureau_balance_aggregated() -> pd.DataFrame:
    """Perform all loading and aggregation from nb5"""
    balance = load_pkl_to_preprocessor("bureau_balance").data

    balance.sort_values(
        ["SK_ID_BUREAU", "MONTHS_BALANCE"], ascending=(True, True), inplace=True
    )

    ## aggregate 4 most recent states
    agg = encode_agg_bureau_balance(
        balance,
        lookback=4,
        forecast=0,
    )
    agg = agg[agg["MONTHS_BALANCE"] == 0].drop(columns=["MONTHS_BALANCE"])

    # Count bad payments
    bad_payments = balance[balance["STATUS"].isin(["1", "2", "3", "4", "5"])]
    bad_payments = (
        bad_payments.groupby("SK_ID_BUREAU").size().reset_index(name="bad_payments")
    )

    agg_bureau_balance = pd.merge(
        agg,
        bad_payments,
        on="SK_ID_BUREAU",
        how="outer",
    )

    agg_bureau_balance["bad_payments"] = (
        agg_bureau_balance["bad_payments"].fillna(0).astype(int)
    )

    return agg_bureau_balance