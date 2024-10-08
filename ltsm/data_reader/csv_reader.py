import pandas as pd
from pathlib import Path
from ltsm.common.base_reader import BaseReader
import warnings

class CSVReader(BaseReader):
    module_id = "csv"
    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def fetch(self) -> pd.DataFrame:
        # input: path
        # output: DataFrame
        if not Path(self.data_path).is_file():
            raise FileNotFoundError(f"File not found at the specified path: {self.data_path}")

        # Read data, extract columns, toss non-datetime columns
        try:
            loaded_data = pd.read_csv(self.data_path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file at {self.data_path} is empty.")
        except pd.errors.ParserError:
            raise ValueError(f"Failed to parse CSV file at {self.data_path}.")
        
        for col in loaded_data.columns:
            if not col.isnumeric() and not self.__is_datetime(col):
                # Drop columns that are either not labeled by a datetime or an index
                warnings.warn(f"Dropping column '{col}' as its column name is neither a number nor a datetime.")
                loaded_data.drop(columns=col, inplace=True)
            elif not pd.api.types.is_float_dtype(loaded_data[col]):
                # Drop columns that do not contain float data
                warnings.warn(f"Dropping column '{col}' as it does not contain float data.")
                loaded_data.drop(columns=col, inplace=True)

        
        # Fill NA through linear interpolation
        def fillna(row):
            if row.isna().any():
                return row.interpolate(method='linear', inplace=False)
            return row

        loaded_data = loaded_data.apply(fillna, axis=1)
        return loaded_data
    
    def __is_datetime(self, label: str) -> bool:
        try:
            pd.to_datetime(label)
            return True
        except ValueError:
            return False