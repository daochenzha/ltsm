import numpy as np
import pandas as pd
import mne, scipy

from tsbench.data_pipeline.reader.base_reader import BaseReader

class OPMReader(BaseReader):
    module_id = "opm"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self):
        df = self._fif_to_dataframe(self.data_path)
        # Remove potentially unused dimensions
        timeseries = [df[i].squeeze().astype(np.float32) for i in range(len(df))]
        return timeseries
    
    # Possible channels: [0, 1, ..., 8]
    def _fif_to_dataframe(self, data_path, incl_channels = [0]):
        # Reading raw data
        raw = mne.io.read_raw(data_path, verbose = "error")
        
        # Finding applicable channels
        picks = mne.pick_types(raw.info, meg=True)
        # Remove unwanted channels beforehand to speed up computations
        remv_picks = np.setdiff1d(picks, incl_channels)
        ch_names = raw.info.ch_names
        raw.drop_channels([ch_names[i] for i in remv_picks])
        
        # Load data and apply preprocessing
        raw.load_data(verbose = "error")
        raw.filter(None, 90, h_trans_bandwidth=10.0, verbose = "error")
        raw.notch_filter(50.0, notch_widths=1, verbose = "error")
        
        # Find events
        events = mne.find_events(raw, stim_channel="STI101", verbose = "error")
        
        # Split data into separate event intervals
        event_id = dict(Median=257)
        picks = mne.pick_types(raw.info, meg=True)
        reject = dict(mag=2e-10)

        epochs = mne.Epochs(
            raw, events, event_id, -0.5, 1, verbose="error", reject=reject,
            picks=picks, proj=False, decim=10, preload=True,
        )

        # Return data
        return epochs._data