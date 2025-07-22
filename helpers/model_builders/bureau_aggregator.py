import pandas as pd
import numpy as np



def simplify_credit_active(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplify the CREDIT_ACTIVE column to only include 'Active', 'Closed', and 'Sold'.
    """
    df = df.copy()
    df.loc[df["CREDIT_ACTIVE"].isin(["Bad debt", "Sold"]), "CREDIT_ACTIVE"] = "Sold"

    

    return df


def choose_currency(df: pd.DataFrame, currency: str = "currency 1") -> pd.DataFrame:
    """Choose only rows with the specified currency."""
    print(f"Filtering for currency: {currency}")

    return df[df['CREDIT_CURRENCY'] == currency]

class BureauAggregator:
    """
    Class to aggregate bureau data by client ID with proper subfunctions for each metric group.
    Output data can be used for scale-independent models
    """

    def __init__(self, bureau_df: pd.DataFrame) -> None:
        df = bureau_df.copy()
        df = simplify_credit_active(df)
        self.bureau_df = choose_currency(df, "currency 1")
        self.grouped = self.groupby_id_and_status()
        self.agg_df = None
        self.final_df = None

    
        


    def groupby_id_and_status(self) -> pd.DataFrame:
        """Groups the data and prepares it for aggregation"""

        self.grouped = self.bureau_df.groupby(
            ["SK_ID_CURR", "CREDIT_ACTIVE"], observed=True
        )
        return self.grouped

    def aggregate_columns(self, columns: list, agg_func: str) -> pd.DataFrame:

        """Generic aggregation method for specified columns and function"""
        if agg_func == "count_rows":
            return self.grouped.size().to_frame(name=columns)

        elif agg_func not in ["sum", "mean", "max", "min"]:
            raise ValueError(
                f"Unsupported aggregation function: {agg_func}."
                + "Supported functions are: 'sum', 'mean', 'max', 'min', 'count_rows'."
            )
        
        return getattr(self.grouped[columns], agg_func)()

    def aggregate_by_dict(self, metrics_dict: dict[str : list[str]]) -> pd.DataFrame:
        """Aggregate the dataframe by the provided metrics dictionary"""
        df = pd.DataFrame()

        for agg_func, columns in metrics_dict.items():

            agg_result = self.aggregate_columns(columns, agg_func)

            agg_result.columns = [agg_func + "_" + col for col in columns]
            df = pd.concat([df, agg_result], axis=1)

        self.agg_df = df

        return self.agg_df

    def flatten_for_ml(self) -> pd.DataFrame:
        """Pivot the aggregated dataframe to have one row per client"""
        df = self.agg_df.copy()

        df = df.unstack(level="CREDIT_ACTIVE").reset_index()
        df.columns = df.columns.map(lambda t: f"{t[1]}_{t[0]}")

        df = df.rename(columns={"_SK_ID_CURR": "SK_ID_CURR"})

        self.final_df = df
        return self.final_df