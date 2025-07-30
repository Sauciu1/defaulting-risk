import pandas as pd
import numpy as np

from .. import preprocessing
from .. import bureau_helper


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
    balance = preprocessing.load_pkl_to_preprocessor("bureau_balance").data

    balance.sort_values(
        ["SK_ID_BUREAU", "MONTHS_BALANCE"], ascending=(True, True), inplace=True
    )

    ## aggregate 4 most recent states
    agg = bureau_helper.encode_agg_bureau_balance(
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

