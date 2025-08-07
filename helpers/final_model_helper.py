from json import load
from typing import Literal
from shap import models
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
import joblib

from .utils import sklearn_helper, aggregator_helper
from . import table_navigator, preprocessing, bureau_helper
from .model_builders import bureau_preparation
import os

import pandas as pd
import numpy as np


class preprocessor_for_sklearn:
    """Allows for the creation of a preprocessor for SKLEARN models.
    This is used to ensure that the same preprocessing steps are applied to both training and deployed model data.
    """

    def __init__(self, columns_path) -> None:
        self.load_columns(columns_path)

    def load_columns(self, columns_dict_path: str) -> dict[str : list[str]]:
        if not os.path.exists(columns_dict_path):
            raise FileNotFoundError(
                f"Columns dictionary not found at: {columns_dict_path}"
            )

        columns_dict = joblib.load(columns_dict_path)

        self.num_cols = columns_dict["num_cols"]
        self.cat_cols = columns_dict["cat_cols"]
        self.bool_cols = columns_dict["bool_cols"]
        self.id_cols = columns_dict["id_cols"]
        self.y_col = columns_dict["y_col"]
        self.X_cols = columns_dict["X_cols"]
        print(f"Loaded X and y columns: {len(self.X_cols)} + 1")

        return columns_dict

    def get_preprocessor(self) -> ColumnTransformer:

        cat_as_obj_transformer = make_column_transformer(
            (FunctionTransformer(lambda x: x.astype("str")), self.cat_cols),
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        cat_as_obj_transformer.set_output(transform="pandas")

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "str_converter",
                                sklearn_helper.category_transformer(
                                    self.cat_cols, "str"
                                ),
                            ),
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    self.cat_cols,
                ),
            ],
            remainder="drop",
        )

        return preprocessor


class tables_to_input_converters:
    """This class is used to get, convert and save the input tables for the final model.
    It abbreviates all actions performed nb3 to nb7 into a single function call."""

    def __init__(self, tables_path: str, columns_schema_path: str) -> None:
        self.tables_path = tables_path
        self.columns_schema_path = columns_schema_path
        self.app = None
        self.bureau_balance = None
        self.bureau_agg = None
        self.prev_app = None

    def get_full_input(self) -> pd.DataFrame:
        """Runs all table aggregations and returns the full input table for the final model."""


        if self.app is None:
            self.app = self.nb1_get_application_table()
        if self.bureau_agg is None:
            self.bureau_agg = self.nb6_bureau_agg()
        if self.prev_app is None:
            self.prev_app = self.nb7_previous_applications()

        full = self.app.merge(self.bureau_agg, on="SK_ID_CURR", how="left")
        full = full.merge(self.prev_app, on="SK_ID_CURR", how="left")
        return full

    def nb1_get_application_table(self) -> pd.DataFrame:
        """Get the application table, which includes both training and test data."""
        app_train = self.nb3_get_input_table("application_train")

        app_test = self.nb3_get_input_table("application_test")

        base = pd.concat([app_train.data, app_test.data], ignore_index=True)
        base["DAYS_EMPLOYED"] = (
            base["DAYS_EMPLOYED"].replace(365243, np.nan).astype(float)
        )
        app_train.data = base

        self.app = app_train.convert_all("category")
        return self.app

    def nb3_get_input_table(self, table: str) -> dict[str, pd.DataFrame]:
        """Get the input tables from the given path."""
        if not os.path.exists(self.tables_path):
            raise FileNotFoundError(f"Tables path not found: {self.tables_path}")
        table = preprocessing.load_pkl_to_preprocessor(
            table,
            tables_path=self.tables_path,
            columns_dict_path=self.columns_schema_path,
        )
        table.convert_all("category")

        return table

    def nb5_bureau_balance_agg(
        self,
    ) -> pd.DataFrame:
        """Get the bureau balance aggregation table. as per workflow of NB5"""
        if self.bureau_balance is not None:
            return self.bureau_balance

        balance = self.nb3_get_input_table("bureau_balance").convert_all("category")

        balance = balance.loc[balance["MONTHS_BALANCE"] > -4, :]

        agg = bureau_helper.encode_agg_bureau_balance(
            data=balance,
            lookback=4,
            forecast=0,
            test=True,
        )

        agg_df = agg[agg["MONTHS_BALANCE"] == 0].drop(columns=["MONTHS_BALANCE"])

        print(f"Relevant bureau balance shape: {agg_df.shape}")

        def include_bad_payments(balance, agg_df) -> pd.DataFrame:
            bad_payments = balance[balance["STATUS"].isin(["2", "3", "4", "5"])]
            bad_payments = (
                bad_payments.groupby("SK_ID_BUREAU")
                .size()
                .reset_index(name="bad_payments")
            )

            agg_bureau_balance = pd.merge(
                agg_df,
                bad_payments,
                on="SK_ID_BUREAU",
                how="outer",
            )
            return agg_bureau_balance

        agg_balance = include_bad_payments(balance, agg_df)
        self.bureau_balance = agg_balance
        return self.bureau_balance

    def nb6_bureau_agg(self) -> pd.DataFrame:
        bureau_df = self.nb3_get_input_table("bureau").convert_all("category")

        # Fix: Add the missing CREDIT_ACTIVE processing

        bureau_df = bureau_preparation.simplify_credit_active(bureau_df)
        bureau_df = bureau_preparation.choose_currency(bureau_df)

        bureau_df["credit_usage"] = (
            bureau_df["AMT_CREDIT_SUM"] / bureau_df["AMT_CREDIT_SUM_LIMIT"]
        ).replace(np.inf, np.nan)
        bureau_df["credit_duration"] = (
            bureau_df["DAYS_CREDIT"] - bureau_df["DAYS_ENDDATE_FACT"]
        )

        bureau_df = bureau_helper.simplify_credit_type(bureau_df)
        
        balance_df = self.nb5_bureau_balance_agg()
        balance_df = bureau_helper.process_bureau_balance(balance_df)
        bureau_df["SK_ID_BUREAU"] = bureau_df["SK_ID_BUREAU"].astype(str)

        bureau_df = pd.merge(bureau_df, balance_df, on="SK_ID_BUREAU", how="left")

        # Debug: Check if CREDIT_ACTIVE exists
        print("Available columns:", bureau_df.columns.tolist())
        print(
            "CREDIT_ACTIVE unique values:",
            (
                bureau_df["CREDIT_ACTIVE"].unique()
                if "CREDIT_ACTIVE" in bureau_df.columns
                else "Column not found"
            ),
        )

        agg = aggregator_helper.RowAggregator(
            bureau_df,
            stack_columns=["SK_ID_CURR", "CREDIT_ACTIVE"],
            agg_dict="auto",
        )
        self.bureau_agg = agg.get_df()
        return self.bureau_agg

    def nb7_previous_applications(self) -> pd.DataFrame:
        prev_app = self.nb3_get_input_table("previous_application").convert_all(
            "category"
        )

        numeric_cols = prev_app.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = prev_app.select_dtypes(include=["category"]).columns.tolist()
        input_dict = {
            **{col: ["sum", "max", "median", "min", "std"] for col in numeric_cols},
            **{col: "onehot_count" for col in cat_cols},
        }

        agg_prev = aggregator_helper.RowAggregator(
            df=prev_app,
            stack_columns=["SK_ID_CURR"],
            agg_dict=input_dict,
        ).get_df()



        subtables = self.merge_previous_suptables()
        subtables.drop(columns=["SK_ID_PREV"], inplace=True, errors="ignore")
        agg_prev.drop(columns="SK_ID_PREV", inplace=True, errors="ignore")

        self.prev_app = pd.merge(
            agg_prev,
            subtables,
            on="SK_ID_CURR",
            how="outer",
        )
        return self.prev_app
    
    
    def merge_previous_suptables(self) -> pd.DataFrame:
        pos_agg_df = self._get_prev_supp_table("POS_CASH_balance")
        installments_agg_df = self._get_prev_supp_table("installments_payments")
        credit_agg_df = self._get_prev_supp_table("credit_card_balance")

        base = pos_agg_df.merge(
            installments_agg_df,
            on="SK_ID_CURR",
            how="outer",
        )
        base = base.merge(
            credit_agg_df,
            on="SK_ID_CURR",
            how="outer",
        )
        return base
    
    def _get_prev_supp_table(self, table : Literal["POS_CASH_balance", "installments_payments", "credit_card_balance"]) -> pd.DataFrame:
        """Get the previous application supplementary table based on the table name."""
        prefixes = {
            "POS_CASH_balance": "pos",
            "installments_payments": "inst",
            "credit_card_balance": "credit",
        }


        _, credit_agg = get_agg_df(
            table,
            group_by="SK_ID_CURR",
            prefix=prefixes[table],
            no_prefix_cols=["SK_ID_CURR"],
)
        return credit_agg

    


def get_agg_df(table:str, group_by:str="SK_ID_CURR", prefix:str=None, no_prefix_cols:list[str]=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Automatically aggregates a dataframe based on the provided table name.
    Returns the raw dataframe and the aggregated dataframe."""
    no_prefix_cols

    raw_df = preprocessing.load_pkl_to_preprocessor(table).convert_all("category")
    agg = aggregator_helper.RowAggregator(
        df=raw_df,
        stack_columns=["SK_ID_CURR"],
        agg_dict='auto',
    )
    agg_df = agg.get_df(prefix=prefix, no_prefix_cols=no_prefix_cols)
    return raw_df, agg_df