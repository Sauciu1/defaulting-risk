from json import load
from shap import models
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
import joblib

from helpers.utils import aggregator_helper
from .utils import sklearn_helper
import os
from . import table_navigator, preprocessing, bureau_helper
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

    def __init__(self, tables_path: str, columns_schema_path:str) -> None:
        self.tables_path = tables_path
        self.columns_schema_path = columns_schema_path

    def get_full_input(self) -> pd.DataFrame:
        """Runs all table aggregations and returns the full input table for the final model."""

        app = self.nb1_get_application_table()
        bureau_agg = self.nb6_bureau_agg()
        prev_app = self.nb7_previous_applications()

        full = app.merge(bureau_agg, on="SK_ID_CURR", how="left")
        full = full.merge(prev_app, on="SK_ID_CURR", how="left")

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
        app_train.data = app_train.convert_all("category")
        base = app_train.data

        return base

    def nb3_get_input_table(self, table: str) -> dict[str, pd.DataFrame]:
        """Get the input tables from the given path."""
        if not os.path.exists(self.tables_path):
            raise FileNotFoundError(f"Tables path not found: {self.tables_path}")
        table = preprocessing.load_pkl_to_preprocessor(
            table, tables_path=self.tables_path, columns_dict_path=self.columns_schema_path
        )
        table.convert_all("category")

        return table

    def nb5_bureau_balance_agg(
        self,
    ) -> pd.DataFrame:
        """Get the bureau balance aggregation table. as per workflow of NB5"""
        balance = self.nb3_get_input_table("bureau_balance").convert_all("category")

        balance= balance.loc[balance["MONTHS_BALANCE"] >= -12, :].copy()

        agg = bureau_helper.encode_agg_bureau_balance(
            data=balance,
            lookback=4,
            forecast=0,
            include_end_month=False,
            X_y_split=False,
            test=True,
        )

        agg_df = agg[agg["MONTHS_BALANCE"] == 0].drop(columns=["MONTHS_BALANCE"])

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
            agg_bureau_balance = include_bad_payments(balance, agg_df)

            agg_bureau_balance["bad_payments"] = (
                agg_bureau_balance["bad_payments"].fillna(0).astype(int)
            )
            return agg_bureau_balance

        agg_balance = include_bad_payments(balance, agg_df)

        return agg_balance

    def nb6_bureau_agg(self):
        bureau_df = self.nb3_get_input_table("bureau").convert_all("category")

        bureau_df["credit_usage"] = (
            bureau_df["AMT_CREDIT_SUM"] / bureau_df["AMT_CREDIT_SUM_LIMIT"]
        ).replace(np.inf, np.nan)
        bureau_df["credit_duration"] = (
            bureau_df["DAYS_CREDIT"] - bureau_df["DAYS_ENDDATE_FACT"]
        )

        metrics_dict = {
            "CREDIT_ACTIVE": "count",
            **{
                col: ["min", "median", "max", "std", "sum"]
                for col in bureau_df.select_dtypes(include=[np.number]).columns
            },
        }
        bureau_df = bureau_helper.simplify_credit_type(bureau_df)
        metrics_dict["CREDIT_TYPE"] = "onehot_count"

        metrics_dict["bur_bal_bad_payments"] = ["sum", "max", "median" "min", "std"]
        metrics_dict["bur_bal_f_0"] = ["onehot_count"]

        balance_df = self.nb5_bureau_balance_agg()

        bureau_df = pd.merge(bureau_df, balance_df, on="SK_ID_BUREAU", how="left")
        agg = aggregator_helper.RowAggregator(
            bureau_df,
            stack_columns=["SK_ID_CURR", "CREDIT_ACTIVE"],
            agg_dict=metrics_dict,
        )

        agg_bureau = agg.aggregate(
            groupby_columns=["SK_ID_CURR"],
            dropna=True,
        )

        return agg_bureau

    def nb7_previous_applications(self) -> pd.DataFrame:
        prev_app = self.nb3_get_input_table("previous_application")

        numeric_cols = prev_app.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = prev_app.select_dtypes(include=["category"]).columns.tolist()
        input_dict = {
            **{col: ["sum", "max", "median" "min", "std"] for col in numeric_cols},
            **{col: "onehot_count" for col in cat_cols},
        }

        agg_prev = aggregator_helper.RowAggregator(
            df=prev_app,
            stack_columns=["SK_ID_CURR"],
            agg_dict=input_dict,
        )

        def get_agg_POS_CASH_balance() -> pd.DataFrame:
            pos = preprocessing.load_pkl_to_preprocessor(
                "POS_CASH_balance"
            ).convert_all("category")

            pos_agg_df = aggregator_helper.RowAggregator(
                df=pos, stack_columns=["SK_ID_CURR"], agg_dict="auto"
            ).get_df()
            return pos_agg_df

        def get_agg_installments() -> pd.DataFrame:
            installments = preprocessing.load_pkl_to_preprocessor(
                "installments_payments"
            ).convert_all("category")

            installments_agg_df = aggregator_helper.RowAggregator(
                df=installments, stack_columns=["SK_ID_CURR"], agg_dict="auto"
            ).get_df()
            return installments_agg_df

        def get_agg_credit_card_balance() -> pd.DataFrame:
            """Get the credit card balance sub-table"""
            credit = preprocessing.load_pkl_to_preprocessor(
                "credit_card_balance"
            ).convert_all("category")

            credit_agg_df = aggregator_helper.RowAggregator(
                df=credit, stack_columns=["SK_ID_CURR"], agg_dict="auto"
            ).get_df()
            return credit_agg_df

        def previous_subtables() -> pd.DataFrame:
            pos_agg_df = get_agg_POS_CASH_balance()
            installments_agg_df = get_agg_installments()
            credit_agg_df = get_agg_credit_card_balance()

            base = prev_app.merge(
                pos_agg_df,
                on="SK_ID_CURR",
                how="full",
            )
            base = base.merge(
                installments_agg_df,
                on="SK_ID_CURR",
                how="full",
            )
            base = base.merge(
                credit_agg_df,
                on="SK_ID_CURR",
                how="full",
            )
            return base

        subtables = previous_subtables()
        prev_app = pd.merge(
            agg_prev.get_df(),
            subtables,
            on="SK_ID_CURR",
            how="full",
        )

        return prev_app
    

