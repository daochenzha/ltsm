import numpy as np
import pandas as pd
import mne, scipy

from tsbench.data_pipeline.reader.base_reader import BaseReader

class KilowordReader(BaseReader):
    module_id = "kiloword"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self):
        df = self._fif_to_dataframe(self.data_path)
        # Remove potentially unused dimensions
        timeseries = [df[i].squeeze().astype(np.float32) for i in range(len(df))]
        return timeseries
    
    # Possible channels: [0, 1, ..., 28]
    def _fif_to_dataframe(self, data_path, incl_channels = [0]):
        # Reading epoched data
        epochs = mne.read_epochs(data_path, verbose = "error")

        # Finding applicable channels
        picks = mne.pick_types(epochs.info, eeg = True)
        # Remove unwanted channels beforehand to speed up computations
        remv_picks = np.setdiff1d(picks, incl_channels)
        ch_names = epochs.info.ch_names
        epochs.drop_channels([ch_names[i] for i in remv_picks])

        # Return data
        return epochs._data