from typing import Literal
import pandas as pd
from sklearn import preprocessing
import os
import pickle

from . import table_navigator
import numpy as np


pd.set_option("future.no_silent_downcasting", True)


def column_list_adapter(columns: list[str] | set[str]) -> list[str]:
    """Convert various types of column inputs to a list of strings."""
    if isinstance(columns, str):
        return [columns]
    elif isinstance(columns, set):

        return list(columns)

    elif isinstance(columns, list):
        return columns
    else:
        raise TypeError(f"Unsupported column type: {type(columns)}")


def convert_bool_to_bool(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert specified columns to boolean type."""
    columns = column_list_adapter(columns)
    df = df.copy()  # Work on a copy to avoid modifying original

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue

        if df[col].isna().any():
            print(
                f"Warning: Column '{col}' contains NaN values. These will become False."
            )

        if df[col].dtype == "object":
            replacement_dict = {
                "Y": True,
                "N": False,
                "Yes": True,
                "No": False,
                "True": True,
                "False": False,
            }
            df[col] = df[col].astype(str).replace(replacement_dict)

        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace({1: True, 0: False, 1.0: True, 0.0: False})

        df[col] = df[col].astype(bool)

    return df


def convert_to_object(df: pd.DataFrame, columns: set) -> pd.DataFrame:
    columns = column_list_adapter(columns)
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
    return df


def convert_to_category(df: pd.DataFrame, columns: set) -> pd.DataFrame:
    columns = column_list_adapter(columns)
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("category")
    return df


class Preprocessor:
    """A class for preprocessing DataFrame columns for ML tasks."""

    def __init__(
        self,
        data: pd.DataFrame,
        cat_columns: list[str] | None,
        num_columns: list[str] | None,
        bool_columns: list[str] | None,
        str_columns: list[str] | None,
    ) -> None:

        self.data = data
        self.cat_columns = column_list_adapter(cat_columns)
        self.num_columns = column_list_adapter(num_columns)
        self.bool_columns = column_list_adapter(bool_columns)
        self.str_columns = column_list_adapter(str_columns)

        if "SK_ID_CURR" in self.data.columns:
            self.data["SK_ID_CURR"] = self.data["SK_ID_CURR"].astype(str)

        self.categorical_type: Literal["object", "category"] = None

        self.column_structure = None

    def convert_all(self, type: Literal["object", "category"] = None) -> pd.DataFrame:
        self.data = convert_bool_to_bool(self.data, self.bool_columns)

        if type is None:
            type = self.categorical_type

        self.categorical_type = type
        if type == "object":
            self.data = convert_to_object(self.data, self.cat_columns)
        elif type == "category":
            self.data = convert_to_category(self.data, self.cat_columns)
        elif type is None:
            pass
        else:
            raise ValueError(f"Unsupported type: {type}. Use 'object' or 'category'.")
        
        self.data = convert_to_object(self.data, self.str_columns)

        self.check_columns()
        return self.data

    def check_columns(self) -> tuple[list[str] | None, list[str] | None]:
        """Check for missing and unexpected columns."""
        missing = self.missing_columns()
        unexpected = self.unexpected_columns()

        if missing or unexpected:
            print(f"Missing columns: {missing}; Unexpected columns: {unexpected}")

        return {"missing": missing, "unexpected": unexpected}

    def _get_provided_columns(self) -> set[str]:
        """Get all columns that should be in the DataFrame."""
        return list(
            self.cat_columns + self.num_columns + self.bool_columns + self.str_columns
        )

    def missing_columns(self) -> list[str] | None:
        """Check for columns that were missing in the df."""
        missing_columns = set(self._get_provided_columns()) - set(self.data.columns)

        return list(missing_columns)

    def unexpected_columns(self) -> list[str] | None:
        """Check for columns that were not expected."""
        unexpected_columns = set(self.data.columns) - set(self._get_provided_columns())
        return list(unexpected_columns)

    def get_expected_df(self, include_extra: list[str] = None) -> pd.DataFrame:
        """Get a DataFrame with only the expected columns."""
        expected_columns = self._get_provided_columns()
        result = self.data[expected_columns + (include_extra or [])].copy()
        if result.empty:
            raise ValueError(
                "No expected columns found in the DataFrame. "
                "Check if the DataFrame contains any of the expected columns."
            )

        return result

    def get_X_y_id(
        self, target_column: str = "TARGET", id_column: str = "SK_ID_CURR"
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Get features and target variable from the DataFrame."""

        X = self.get_expected_df()
        if target_column in X.columns:
            X = X.drop(columns=[target_column])
        if id_column in X.columns:
            X = X.drop(columns=[id_column])

        y = self.data[target_column] if target_column in self.data.columns else None
        id = self.data[id_column] if id_column in self.data.columns else None

        print(
            f"X shape: {X.shape}, y shape: {y.shape if y is not None else 'None'}, id shape: {id.shape if id is not None else 'None'}"
        )

        return X, y, id

    def check_conversion_success(self) -> pd.DataFrame | None:
        """Check if the DataFrame has the expected columns after conversion."""
        type_checks = {
            "category": lambda s: s.dtype.name == self.categorical_type,
            "numeric": pd.api.types.is_numeric_dtype,
            "bool": lambda s: s.dtype.name == "bool",
            "object": lambda s: s.dtype.name == "object",
        }

        type_map = {
            "category": self.cat_columns,
            "numeric": self.num_columns,
            "bool": self.bool_columns,
            "object": self.str_columns,
        }

        incorrect_types = []
        for expected_type, columns in type_map.items():
            for col in columns:
                check_func = type_checks[expected_type]

                if not check_func(self.data[col]):
                    incorrect_types.append(
                        {
                            "column": col,
                            "actual_dtype": self.data[col].dtype,
                            "expected_type": expected_type,
                        }
                    )

        if incorrect_types:
            print("Columns with incorrect data types found and returned")
            df_incorrect = pd.DataFrame(incorrect_types)
            return df_incorrect

        else:
            print("All columns have the expected data types.")
        return None


def load_pkl_to_preprocessor(
    table: Literal[
        "application_train",
        "application_test",
        "bureau",
        "bureau_balance",
        "credit_card_balance",
        "POS_CASH_balance",
        "previous_application",
        "installments_payments",
    ],
    tables_path: str = "data/raw_csv",
    columns_dict_path: str = "data/processed"
) -> Preprocessor:
    """Construct preprocessor anew using pickled files and raw_data."""

    df = table_navigator.get_tables_from_dir(tables_path, table)[table]

    if table in ["application_train", "application_test"]:
        with open(os.path.join(columns_dict_path, "application_columns.pkl"), "rb") as f:
            pkl_columns = pickle.load(f)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan).astype(float)

    elif table in [
        "bureau",
        "bureau_balance",
        "credit_card_balance",
        "POS_CASH_balance",
        "previous_application",
        "installments_payments",
    ]:
        with open(os.path.join(columns_dict_path, "supplementary_tables_columns.pkl"), "rb") as f:
            pkl_columns = pickle.load(f)
    else:
        raise ValueError(f"Unsupported table: {table}.")

    col_types = ["cat_columns", "num_columns", "bool_columns", "str_columns"]

    columns = {
        col_type: column_list_adapter(
            [col for col in pkl_columns[col_type] if col in df.columns]
        )
        for col_type in col_types
    }


    preprocessor = Preprocessor(
        data=df,
        cat_columns=columns["cat_columns"],
        num_columns=columns["num_columns"],
        bool_columns=columns["bool_columns"],
        str_columns=columns["str_columns"],
    )

    if table in ["application_train", "application_test"]:
        structure = pkl_columns["application_columns"]
    else:
        print(pkl_columns["tables_columns"].keys())
        structure = pkl_columns["tables_columns"][table.lower()]

    if type(structure) is dict:
        preprocessor.column_structure = {
            key: column_list_adapter(value) for key, value in structure.items()
        }
    elif type(structure) is list:
        preprocessor.column_structure = column_list_adapter(structure)

    preprocessor.convert_all('category')

    return preprocessor
