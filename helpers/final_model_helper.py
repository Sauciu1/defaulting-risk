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

        categorical_transformer = Pipeline(
            [
                (
                    "str_converter",
                    sklearn_helper.category_transformer(self.cat_cols, "str"),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_transformer,
                    self.num_cols,
                ),
                (
                    "cat",
                    categorical_transformer,
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

        self.credit_card_balance = None
        self.installments_payments = None
        self.POS_CASH_balance = None

    def get_full_input(self) -> pd.DataFrame:
        """Efficiently merge all tables with progress tracking."""
        print("Loading application data...")
        if self.app is None:
            self.app = self.nb1_get_application_table()

        print("Processing bureau data...")
        if self.bureau_agg is None:
            self.bureau_agg = self.nb6_bureau_agg()

        print("Processing previous applications...")
        if self.prev_app is None:
            self.prev_app = self.nb7_previous_applications()

        print("Merging tables...")
        full_df = self.app.merge(self.bureau_agg, on="SK_ID_CURR", how="left").merge(
            self.prev_app, on="SK_ID_CURR", how="left"
        )

        print(f"Final dataset shape: {full_df.shape}")
        return full_df

    def nb1_get_application_table(self) -> pd.DataFrame:
        """Get combined application table (train + test)."""
        app_train = self.nb3_get_input_table("application_train")
        app_test = self.nb3_get_input_table("application_test")

        # Combine data properly
        combined_data = pd.concat([app_train.data, app_test.data], ignore_index=True)
        combined_data["DAYS_EMPLOYED"] = (
            combined_data["DAYS_EMPLOYED"].replace(365243, np.nan).astype(float)
        )

        # Create new preprocessor with combined data
        app_train.data = combined_data
        self.app = app_train.convert_all("category")
        return self.app

    def nb3_get_input_table(self, table: str) -> preprocessing.Preprocessor:
        if not os.path.exists(self.tables_path):
            raise FileNotFoundError(f"Tables path not found: {self.tables_path}")

        # Validate specific table exists
        table_path = os.path.join(self.tables_path, f"{table}.csv")
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"Table file not found: {table_path}")

        return preprocessing.load_pkl_to_preprocessor(
            table,
            tables_path=self.tables_path,
            columns_dict_path=self.columns_schema_path,
        )

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
        self.bureau_balance = bureau_helper.process_bureau_balance(agg_balance)
        return self.bureau_balance

    def nb6_bureau_agg(self) -> pd.DataFrame:
        bureau_df = self.nb3_get_input_table("bureau").convert_all("category")
        bureau_df = bureau_helper.simplify_credit_type(bureau_df)
        bureau_df = bureau_helper.credit_feature_engineering(bureau_df)

        balance_df = self.nb5_bureau_balance_agg()

        merged_df = bureau_df.merge(balance_df, on="SK_ID_BUREAU", how="left")

        self.bureau_agg = aggregator_helper.RowAggregator(
            merged_df,
            stack_columns=["SK_ID_CURR", "CREDIT_ACTIVE"],
            agg_dict="auto",
        ).get_df()

        return self.bureau_agg

    def _agg_prev_app_table(self) -> pd.DataFrame:
        _, agg_prev_app = get_agg_df(
            table="previous_application",
            group_by=["SK_ID_CURR"],
            no_prefix_cols=["SK_ID_CURR"],
        )
        return agg_prev_app

    def nb7_previous_applications(self) -> pd.DataFrame:
        """returns self.prev_app"""

        subtables = self.merge_previous_suptables()
        subtables.drop(columns=["SK_ID_PREV"], inplace=True, errors="ignore")

        agg_prev = self._agg_prev_app_table()
        agg_prev.drop(columns=["SK_ID_PREV"], inplace=True, errors="ignore")

        self.prev_app = agg_prev.merge(
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
        ).merge(
            credit_agg_df,
            on="SK_ID_CURR",
            how="outer",
        )
        return base

    def _get_prev_supp_table(
        self,
        table: Literal[
            "POS_CASH_balance", "installments_payments", "credit_card_balance"
        ],
    ) -> pd.DataFrame:
        """Get the previous application supplementary table based on the table name."""
        prefixes = {
            "POS_CASH_balance": "pos",
            "installments_payments": "inst",
            "credit_card_balance": "credit",
        }

        if getattr(self, table) is not None:
            return getattr(self, table)

        _, agg_table = get_agg_df(
            table,
            group_by=["SK_ID_CURR"],
            prefix=prefixes[table],
            no_prefix_cols=["SK_ID_CURR"],
        )

        setattr(self, table, agg_table)

        return agg_table

    def keep_model_columns(
        self, model_column_path="models/model_columns.jbl"
    ) -> pd.DataFrame:
        """Keeps only the columns that are present in the model column path."""
        if self.full_df is None:
            self.get_full_input()
        column_dict = joblib.load(model_column_path)
        model_columns = (
            column_dict["X_cols"] + column_dict["y_col"] + column_dict["id_cols"]
        )
        print(
            f"Keeping {len(model_columns)} columns for the model. From {len(self.full_df.columns)} total columns."
        )

        triaged_df = self.full_df[model_columns]
        return triaged_df


def get_agg_df(
    table: str,
    group_by: list[str] = ["SK_ID_CURR"],
    prefix: str = None,
    no_prefix_cols: list[str] = [],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Automatically aggregates a dataframe based on the provided table name.
    Returns the raw dataframe and the aggregated dataframe."""
    no_prefix_cols

    raw_df = preprocessing.load_pkl_to_preprocessor(
        table, tables_path="data/raw_csv/", columns_dict_path="models"
    ).convert_all("category")

    agg = aggregator_helper.RowAggregator(
        df=raw_df,
        stack_columns=group_by,
        agg_dict="auto",
    )
    agg_df = agg.get_df(prefix=prefix, no_prefix_cols=no_prefix_cols)
    return raw_df, agg_df
