from json import load
from shap import models
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
import joblib
from .utils import sklearn_helper
import os
from . import table_navigator, preprocessing, bureau_helper
import pandas as pd



class preprocessor_for_sklearn:
    """Allows for the creation of a preprocessor for SKLEARN models.
    This is used to ensure that the same preprocessing steps are applied to both training and deployed model data."""
    def __init__(self, columns_path) -> None:
        self.load_columns(columns_path)


    def load_columns(self, columns_dict_path: str) -> dict[str: list[str]]:
        if not os.path.exists(columns_dict_path):
            raise FileNotFoundError(f"Columns dictionary not found at: {columns_dict_path}")

        columns_dict = joblib.load(columns_dict_path)


        self.num_cols = columns_dict['num_cols']
        self.cat_cols = columns_dict['cat_cols']
        self.bool_cols = columns_dict['bool_cols']
        self.id_cols = columns_dict['id_cols']
        self.y_col = columns_dict['y_col']
        self.X_cols = columns_dict['X_cols']
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
                            ("str_converter", sklearn_helper.category_transformer(self.cat_cols, "str")),
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
    def __init__(self, tables_path: str) -> None:
        self.tables_path = tables_path

    def nb3_get_input_table(self, table:str) -> dict[str, pd.DataFrame]:
        """Get the input tables from the given path."""
        if not os.path.exists(self.tables_path):
            raise FileNotFoundError(f"Tables path not found: {self.tables_path}")
        table = preprocessing.load_pkl_to_preprocessor(table,
                                        tables_path=self.tables_path, 
                                        columns_dict_path=self.tables_path)
        return table

    def nb5_bureau_balance_agg(self, )->pd.DataFrame:
        """Get the bureau balance aggregation table. as per workflow of NB5"""
        balance = self.nb3_get_input_table("bureau_balance")

    
        agg  = bureau_helper.encode_agg_bureau_balance(
            data=balance,
            lookback=4,
            forecast=0,
            include_end_month=False,
            X_y_split=False,
            test=True,
        )

        agg_df = agg[agg["MONTHS_BALANCE"] == 0].drop(columns=["MONTHS_BALANCE"])


        def include_bad_payments(balance, agg_df):
            bad_payments = balance[balance["STATUS"].isin(["2", "3", "4", "5"])]
            bad_payments = bad_payments.groupby("SK_ID_BUREAU").size().reset_index(name='bad_payments')

            agg_bureau_balance = pd.merge(
            agg_df,
            bad_payments,
            on="SK_ID_BUREAU",
            how="outer",
            )
            agg_bureau_balance = bad_payments(balance, agg_df)

            agg_bureau_balance["bad_payments"] = agg_bureau_balance["bad_payments"].fillna(0).astype(int
        )
            return agg_bureau_balance
        
        agg_balance = include_bad_payments(balance, agg_df)


        return agg_balance

    def nb6_bureau_agg(self):
        None

