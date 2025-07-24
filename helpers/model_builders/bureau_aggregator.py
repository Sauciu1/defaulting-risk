import pandas as pd
import numpy as np




def simplify_credit_active(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplify the CREDIT_ACTIVE column to only include 'Active', 'Closed', and 'Sold'.
    """
    df = df.copy()
    df["CREDIT_ACTIVE"] = df["CREDIT_ACTIVE"].astype("category").cat.set_categories(['Active', 'Closed', 'Other'])
    df.loc[~df["CREDIT_ACTIVE"].isin(['Active', 'Closed']), "CREDIT_ACTIVE"] = "Other"
    

    

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

    def __init__(self, bureau_df: pd.DataFrame, stack_columns =["SK_ID_CURR", "CREDIT_ACTIVE"]) -> None:
        df = bureau_df.copy()
        df = simplify_credit_active(df)
        self.bureau_df = choose_currency(df, "currency 1")
        self.stack_columns = stack_columns
        self.grouped = self.groupby_stack_columns(stack_columns)
        self.agg_df = None
        self.final_df = None


    def groupby_stack_columns(self, stack_columns: list[str]) -> pd.DataFrame:
        """Groups the data and prepares it for aggregation"""

        self.grouped = self.bureau_df.groupby(
            stack_columns, observed=True
        )
        return self.grouped

    def aggregate_columns(self, columns: list, agg_func: str) -> pd.DataFrame:

        """Generic aggregation method for specified columns and function"""
        supported = ["sum", "mean", "max", "min", "count", 'median']

        if agg_func not in supported:
            raise ValueError(
                f"Unsupported aggregation function: {agg_func} not in {supported}",
            )
        
        return getattr(self.grouped[columns], agg_func)()

    def aggregate_by_dict(self, metrics_dict: dict[str : list[str]]) -> pd.DataFrame:
        """Aggregate the dataframe by the provided metrics dictionary"""
        agg_results = []

        for column, agg_funcs in metrics_dict.items():
            if isinstance(agg_funcs, str):
                agg_funcs = [agg_funcs]
            for agg_func in agg_funcs:
                result = self.aggregate_columns([column], agg_func)
                result.columns = [f"{agg_func}_{column}"]
                agg_results.append(result)

        self.agg_df = pd.concat(agg_results, axis=1)
        return self.agg_df

    def unstack_df(self, unstack_columns: list[str]=None) -> pd.DataFrame:
        """Pivot the aggregated dataframe to have one row per client"""
        if unstack_columns is None:
            unstack_columns = self.stack_columns[1:]

            if len(self.stack_columns) == 0:
                return self.agg_df

            


        df = self.agg_df.copy()

        df = df.unstack(level=unstack_columns).reset_index()
        
        name_func = lambda t: "_".join(str(x) for x in t if x != "") if isinstance(t, tuple) else str(t)

        df.columns = df.columns.map(name_func)

        df.columns = [col.strip("_") for col in df.columns]

        self.final_df = df
        return self.final_df