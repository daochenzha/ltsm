import numpy as np
import pandas as pd
import mne, scipy

from tsbench.data_pipeline.reader.base_reader import BaseReader

class SSVEPReader(BaseReader):
    module_id = "ssvep"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self):
        df = self._brainV_to_dataframe(self.data_path)
        # Remove potentially unused dimensions
        timeseries = [df[i].squeeze().astype(np.float32) for i in range(len(df))]
        return timeseries
    
    # Possible channels: [0, 1, ..., 31]
    def _brainV_to_dataframe(self, data_path, incl_channels = [0]):
        # Reading raw data
        raw = mne.io.read_raw_brainvision(data_path, verbose = "error")
        
        # Finding applicable channels
        picks = mne.pick_types(raw.info, eeg=True)
        # Remove unwanted channels beforehand to speed up computations
        remv_picks = np.setdiff1d(picks, incl_channels)
        ch_names = raw.info.ch_names
        raw.drop_channels([ch_names[i] for i in remv_picks])
        
        # Load data and apply preprocessing
        raw.load_data(verbose = "error")
        raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin", verbose=False)
        raw.notch_filter(50.0, notch_widths=1, verbose = "error")
        
        # Find events
        if (len(incl_channels) > 1):
            raw.set_eeg_reference("average", projection=False, verbose=False)
        events, _ = mne.events_from_annotations(raw, verbose = "error")
        
        # Split data into separate event intervals
        event_id = {"12hz": 255, "15hz": 155}
        picks = mne.pick_types(raw.info, eeg=True)

        epochs = mne.Epochs(
            raw, events, event_id, -0.1, 20, verbose="error", baseline = None,
            proj=False, decim=1, preload=True,
        )

        # Apply further preprocessing
        epochs.resample(50.0, npad = "auto", verbose = "error")

        # Return data
        return epochs._data
