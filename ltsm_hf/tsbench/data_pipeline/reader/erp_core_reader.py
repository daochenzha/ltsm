import numpy as np
import pandas as pd
import mne, scipy

from tsbench.data_pipeline.reader.base_reader import BaseReader

class ERP_CoreReader(BaseReader):
    module_id = "erp_core"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self):
        df = self._fif_to_dataframe(self.data_path)
        # Remove potentially unused dimensions
        timeseries = [df[i].squeeze().astype(np.float32) for i in range(len(df))]
        return timeseries
    
    # Possible channels: [0, 1, ..., 29]
    def _fif_to_dataframe(self, data_path, incl_channels = [0]):
        # Reading raw data
        raw = mne.io.read_raw(data_path, verbose = "error")
        
        # Finding applicable channels
        picks = mne.pick_types(raw.info, eeg=True, eog = True)
        # Remove unwanted channels beforehand to speed up computations
        remv_picks = np.setdiff1d(picks, incl_channels)
        ch_names = raw.info.ch_names
        raw.drop_channels([ch_names[i] for i in remv_picks])
        
        # Load data and apply preprocessing
        raw.load_data(verbose = "error")
        raw.filter(l_freq=0.1, h_freq=40, verbose = "error")
        
        # Find events
        events, _ = mne.events_from_annotations(raw, verbose = "error")
        
        # Split data into separate event intervals
        event_id = {'stimulus/compatible/target_left': 3,
                     'stimulus/compatible/target_right': 4}
        picks = mne.pick_types(raw.info, eeg=True, eog = True)
        reject = {"eeg": 250e-6}

        epochs = mne.Epochs(
            raw, events, event_id, -0.1, 0.4, verbose="error", reject=reject,
            proj=False, decim=10, preload=True,
        )

        # Return data
        return epochs._data