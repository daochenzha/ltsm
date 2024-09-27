from ltsm.data_provider.dataset import TSDataset
from pathlib import Path

import numpy as np
import pandas as pd
from distutils.util import strtobool
from datetime import datetime

from ltsm.common.base_reader import BaseReader

class CSVReader(BaseReader):
    module_id = "csv"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self) -> pd.DataFrame:
        # input: path
        # output: DataFrame

        # Read data, extract columns, toss non-datetime columns
        loaded_data = pd.read_csv(self.data_path)
        for col in loaded_data.columns:
            # Drop columns that are either not labeled by a datetime or an index
            if not col.isnumeric() and not self._is_datetime(col):
                loaded_data.drop(columns=col, inplace=True)
        
        # Fill NA through linear interpolation
        def fillna(row):
            if row.isna().any():
                row.interpolate(method='linear', inplace=True)
            return row

        loaded_data = loaded_data.apply(fillna, axis=1)

        return loaded_data
    
    def _is_datetime(self, label: str) -> bool:
        try:
            pd.to_datetime(label)
            return True
        except ValueError:
            return False