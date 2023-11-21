import numpy as np
import pandas as pd
import mne, scipy

from tsbench.data_pipeline.reader.base_reader import BaseReader

class Hf_SefReader(BaseReader):
    module_id = "hf_sef"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self):
        df = self._fif_to_dataframe(self.data_path)
        # Remove potentially unused dimensions
        timeseries = [df[i].squeeze().astype(np.float32) for i in range(len(df))]
        return timeseries
    
    # Possible channels: [0, 1, ..., 305]
    def _fif_to_dataframe(self, data_path, incl_channels = [0]):
        # Reading raw data
        raw = mne.io.read_raw(data_path, verbose = "error")
        
        # Load data and apply preprocessing
        raw.load_data(verbose = "error")
        
        # Find events
        events = mne.find_events(raw, verbose = "error")
        
        # Split data into separate event intervals
        event_id = dict(hf=1)
        picks = mne.pick_types(raw.info, meg=True)
        epochs = mne.Epochs(
            raw, events, event_id, -0.05, 0.3, verbose="error",
            picks=picks, proj = False, decim=10, preload=True,
        )

        # Due to data formatting, all data must be loaded
        # with desired channels selected afters
        picks = mne.pick_types(raw.info, meg = True)
        # Remove unwanted channels
        remv_picks = np.setdiff1d(picks, incl_channels)
        ch_names = raw.info.ch_names
        epochs.drop_channels([ch_names[i] for i in remv_picks])
        
        # Return data
        return epochs._data
