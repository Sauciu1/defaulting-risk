from inspect import stack
import pandas as pd
import numpy as np



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

    def __init__(self, df: pd.DataFrame, stack_columns: list[str], agg_dict: dict[str, list[str]]) -> None:
        self.df = df.copy()
        self.stack_columns = stack_columns
        self.grouped = self.groupby_stack_columns(stack_columns)
        self.agg_df = self.aggregate_by_dict(agg_dict)
        self.final_df = self.unstack_df()

    def get_df(self) -> pd.DataFrame:
        """Returns the final aggregated dataframe"""
        return self.final_df

    def groupby_stack_columns(self, stack_columns: list[str]) -> pd.DataFrame:
        """Groups the data and prepares it for aggregation"""

        self.grouped = self.df.groupby(
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

    def aggregate_by_dict(self, metrics_dict: dict[str, list[str]]) -> pd.DataFrame:
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

        def name_func(t) -> str:
            """Join unstacked names for columns"""
            if isinstance(t, tuple):
                return "_".join(str(x) for x in t if x != "")
            return str(t)

        df.columns = df.columns.map(name_func)

        df.columns = [col.strip("_") for col in df.columns]

        self.final_df = df
        return self.final_df