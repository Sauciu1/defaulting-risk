import pandas as pd

class ColumnDescriber:
    def __init__(self, tables: dict[str, pd.DataFrame], only_tables: list[str] = None):
        """
        Initialize with a dictionary of tables.
        """
        # Remove .csv from table keys
        self.tables = {k.replace('.csv', ''): v for k, v in tables.items()}


        self.only_tables = only_tables if only_tables is not None else list(self.tables.keys())
        if isinstance(self.only_tables, str):
            self.only_tables = [self.only_tables]


        # Fix the descriptions DataFrame
        self.descriptions = tables["HomeCredit_columns_description"].copy()
        self.descriptions.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        # Remove .csv from Table column
        self.descriptions["Table"] = (
            self.descriptions["Table"]
            .replace({"application_{train|test}.csv": "application_train.csv"})
            .str.replace(".csv", "", regex=False)
        )

    def _get_col_stats(self, row):
        """Get statistics for a specific column."""
        table_name = row['Table']
        col = row['Row']

        if table_name in self.tables and col in self.tables[table_name].columns:
            table = self.tables[table_name]
            non_na = table[col].dropna()
            return {
                'nan_count': table[col].isna().sum(),
                'unique_values': table[col].nunique(),
                'dtype': table[col].dtype,
                'sample_values': non_na.unique()[:min(3, len(non_na))].tolist() if len(non_na) > 0 else []
            }
        else:
            return {
                'nan_n': None,
                'unique_values': None,
                'dtype': None,
                'sample_values': None
            }

    def describe(self, columns: list[str]) -> pd.DataFrame:
        """
        Returns a DataFrame with the meanings and statistics of the columns.
        """
        if not isinstance(columns, list):
            if isinstance(columns, str):
                columns = [columns]
            else:
                raise ValueError("The 'columns' parameter must be a list of column names.")

        desc = self.descriptions.copy()
        df = desc[(desc.Row.isin(columns)) & (desc.Table.isin(self.only_tables))].copy()

        stats = df.apply(self._get_col_stats, axis=1)
        df['nan_n'] = stats.apply(lambda x: x['nan_count'])
        df['unique_vals'] = stats.apply(lambda x: x['unique_values'])
        df['dtype'] = stats.apply(lambda x: x['dtype'])
        df['sample_vals'] = stats.apply(lambda x: x['sample_values'])


        return df
