import numpy as np
import pandas as pd
import mne, scipy

from tsbench.data_pipeline.reader.base_reader import BaseReader

class MTRFReader(BaseReader):
    module_id = "mtrf"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self):
        timeseries = self._mat_to_dataframe(self.data_path).astype(np.float32)
        return timeseries
    
    # Possible channels: [0, 1, ..., 127]
    def _mat_to_dataframe(self, data_path, values = "EEG", incl_channels = [0]):
        # Select relevant data
        data_full = scipy.io.loadmat(data_path)
        if values in data_full.keys():
            vals = data_full[values].T
        else:
            raise Exception("Desired value is missing")

        # Select specific channels
        data = [vals[i] for i in incl_channels]

        # Apply preprocessing
        data = mne.filter.resample(data, down=2, npad="auto")

        # Return data
        return data
