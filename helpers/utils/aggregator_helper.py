import pandas as pd
import numpy as np
import re
from typing import Literal


class RowAggregator:
    """
    Class to automatically run row aggregations on a dataframe.
    Output data can be used for scale-independent models
    inputL
    df: DataFrame to be aggregated
    stack_columns: list of columns to group by e.g. ["SK_ID_CURR", "CREDIT_ACTIVE"]
    aggregate_by_dict: dictionary of columns to aggregate and their aggregation functions
    e.g. {"AMT_CREDIT": ["sum", "mean"], "DAYS_CREDIT": "max"}
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stack_columns: list[str],
        agg_dict: Literal["auto"] | dict[str, list[str]],
    ) -> None:
        self.df = df.copy()
        self.stack_columns = stack_columns
        self.grouped = self.groupby_stack_columns(stack_columns)

        if isinstance(agg_dict, dict):
            agg_dict = agg_dict
        elif agg_dict.lower() == "auto":
            agg_dict = self.generate_agg_dict()
        else:
            raise ValueError("agg_dict must be 'auto' or a dictionary of aggregations")

        self.agg_df = self.aggregate_by_dict(agg_dict)
        self.final_df = self.unstack_df()

    def generate_agg_dict(self):
        """Generates an aggregation dictionary based on numeric and categorical columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=["category"]).columns.tolist()

        agg_dict = {
            **{col: ["sum", "max", "median", "min", "std"] for col in numeric_cols},
            **{col: "onehot_count" for col in cat_cols},
        }

        # Remove stack columns from aggregation
        for col in self.stack_columns[1:]:
            agg_dict[col] = "count"

        return agg_dict

    def get_df(self, prefix:str=None, no_prefix_cols:list[str]=None) -> pd.DataFrame:
        """Returns the final aggregated dataframe"""
        df = self.final_df.copy()
        if prefix:
            df.columns = [col if col in no_prefix_cols else f"{prefix}_{col}" for col in df.columns]

        return df

    def groupby_stack_columns(self, stack_columns: list[str]) -> pd.DataFrame:
        """Groups the data and prepares it for aggregation"""

        self.grouped = self.df.groupby(stack_columns, observed=True)
        return self.grouped

    def onehot_count(self, column: str) -> pd.DataFrame:
        """One-hot encoding for categorical columns"""
        df = self.df[self.stack_columns + column].copy()
        # encoded = df.groupby(self.stack_columns + column)[column].value_counts()
        df = df.pivot_table(
            index=self.stack_columns,
            columns=column,
            aggfunc="size",
            fill_value=0,
            observed=False,
        )
        df.columns = [
            f"onehot_count_{column[0]}_{str(val[0])}".replace(" ", "_")
            for val in df.columns
        ]

        # Remove JSON-like characters from column names
        df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

        return df

    def aggregate_columns(self, column: list, agg_func: str) -> pd.DataFrame:

        # print(column, agg_func)
        """Generic aggregation method for specified columns and function"""
        supported = [
            "sum",
            "mean",
            "max",
            "min",
            "count",
            "median",
            "onehot_count",
            "std",
        ]

        if agg_func not in supported:
            raise ValueError(
                f"Unsupported aggregation function: {agg_func} not in {supported}",
            )
        if agg_func == "onehot_count":
            # One-hot encoding for categorical columns
            return self.onehot_count(column)

        new_df = getattr(self.grouped[column], agg_func)()
        new_df.columns = [f"{agg_func}_{column[0]}"]
        return new_df

    def aggregate_by_dict(self, metrics_dict: dict[str, list[str]]) -> pd.DataFrame:
        """Aggregate the dataframe by the provided metrics dictionary"""
        agg_results = []

        for column, agg_funcs in metrics_dict.items():
            if isinstance(agg_funcs, str):
                agg_funcs = [agg_funcs]
            for agg_func in agg_funcs:
                result = self.aggregate_columns([column], agg_func)
                agg_results.append(result)

        # Fix: Use join='outer' and sort=False to handle index misalignment
        self.agg_df = pd.concat(agg_results, axis=1, join="outer", sort=False)

        # Remove any duplicate columns that might still exist
        self.agg_df = self.agg_df.loc[:, ~self.agg_df.columns.duplicated(keep="first")]

        return self.agg_df

    def unstack_df(self, unstack_columns: list[str] = None) -> pd.DataFrame:
        """Pivot the aggregated dataframe to have one row per client"""
        if unstack_columns is None:
            unstack_columns = self.stack_columns[1:]

            if len(self.stack_columns) == 0:
                return self.agg_df

        df = self.agg_df.copy()
        df = df.unstack(level=unstack_columns).reset_index()

        df.columns = df.columns.map(name_func)

        df.columns = [col.strip("_") for col in df.columns]

        self.final_df = df
        return self.final_df


def name_func(t) -> str:
    """Join unstacked names for columns"""
    if isinstance(t, tuple):
        # Filter out empty strings and join properly
        parts = [str(x) for x in t if str(x) != "" and str(x) != "nan"]
        return "_".join(parts)
    return str(t)
