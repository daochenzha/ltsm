import numpy as np
import pandas as pd
import mne, scipy

from tsbench.data_pipeline.reader.base_reader import BaseReader

class SPMReader(BaseReader):
    module_id = "spm"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self):
        df = self._ctf_to_dataframe(self.data_path)
        # Remove potentially unused dimensions
        timeseries = [df[i].squeeze().astype(np.float32) for i in range(len(df))]
        return timeseries
    
    # Possible channels: [32, 33, ..., 305] (3 ... 31 require compensation channels)
    def _ctf_to_dataframe(self, data_path, incl_channels = [32]):
        # Reading raw data
        raw = mne.io.read_raw_ctf(data_path, verbose = "error")

        # Finding applicable channels
        picks = mne.pick_types(raw.info, meg=True, eeg=False)
        
        # Remove unwanted channels beforehand to speed up computations
        remv_picks = np.setdiff1d(picks, incl_channels)
        ch_names = raw.info.ch_names
        raw.drop_channels([ch_names[i] for i in remv_picks])
        # Load data and apply preprocessing
        raw.load_data(verbose = "error")
        raw.filter(1, 30, method="fir", fir_design="firwin", verbose = "error")

        # Find events
        events = mne.find_events(raw, verbose = "error")

        # Split data into separate event intervals
        event_id = {"faces": 1, "scrambled": 2}
        picks = mne.pick_types(raw.info, meg=True)
        reject = dict(mag=5e-12)

        epochs = mne.Epochs(
            raw, events, event_id, -0.2, 0.6, verbose="error", reject=reject,
            picks=picks, proj=False, preload=True,
        )

        # Apply further preprocessing
        epochs.resample(120.0, npad = "auto", verbose = "error")
        
        # Return data
        return epochs._data