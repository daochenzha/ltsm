import numpy as np
import pandas as pd
import mne, scipy

from tsbench.data_pipeline.reader.base_reader import BaseReader

class SomatoReader(BaseReader):
    module_id = "somato"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self):
        df = self._fif_to_dataframe(self.data_path)
        # Remove potentially unused dimensions
        timeseries = [df[i].squeeze().astype(np.float32) for i in range(len(df))]
        return timeseries

    # Possible channels: [0, 1, ..., 304, 315]
    def _fif_to_dataframe(self, data_path, incl_channels = [0]):
        # Reading raw data
        raw = mne.io.read_raw(data_path, verbose = "error")
        
        # Finding applicable channels
        picks = mne.pick_types(raw.info, eog = True, meg = "grad")
        # Remove unwanted channels beforehand to speed up computations
        remv_picks = np.setdiff1d(picks, incl_channels)
        ch_names = raw.info.ch_names
        raw.drop_channels([ch_names[i] for i in remv_picks])
        
        # Load data and apply preprocessing
        raw.load_data(verbose = "error")
        raw.resample(200, verbose = "error")
        raw.notch_filter(50.0, notch_widths=1, verbose = "error")
        
        # Find events
        events = mne.find_events(raw, verbose = "error")
        
        # Split data into separate event intervals
        event_id = [1]
        picks = mne.pick_types(raw.info, eog = True, meg = "grad")
        reject = {}
        # Including reject criteria if they are necessary
        if (len(np.intersect1d(incl_channels, range(305)))):
            reject["grad"] = 4000e-13
        if (315 in incl_channels):
            reject["eog"] = 350e-6

        epochs = mne.Epochs(
            raw, events, event_id, -1, 3, verbose="error", reject=reject,
            picks=picks, proj=False, decim=10, preload=True,
        )
        
        # Return data
        return epochs._data
    