import numpy as np
import pandas as pd
import mne, scipy

from tsbench.data_pipeline.reader.base_reader import BaseReader

class OPM_AuditoryReader(BaseReader):
    module_id = "opm_auditory"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self):
        df = self._fil_to_dataframe(self.data_path)
        # Remove potentially unused dimensions
        timeseries = [df[i].squeeze().astype(np.float32) for i in range(len(df))]
        return timeseries

    # Possible channels: [0, 1, ..., 93]
    def _fil_to_dataframe(self, data_path, incl_channels = [0]):
        # Reading raw data
        raw = mne.io.read_raw_fil(data_path, verbose="error")

        # Finding applicable channels
        picks = mne.pick_types(raw.info, meg=True)
        # Remove unwanted channels beforehand to speed up computations
        remv_picks = np.setdiff1d(picks, incl_channels)
        ch_names = raw.info.ch_names
        raw.drop_channels([ch_names[i] for i in remv_picks])
        
        # Load data
        raw.load_data(verbose = "error")

        # Find events
        events = mne.find_events(raw, min_duration = 0.1, verbose = "error")

        # Split data into separate event intervals
        picks = mne.pick_types(raw.info, meg=True)

        epochs = mne.Epochs(
            raw, events, tmin = -0.1, tmax = 0.4, verbose="error", picks=picks, 
            baseline = (-0.1, 0), decim=10, preload=True,
        )

        # Return data
        return epochs._data