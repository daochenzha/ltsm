import numpy as np
import pandas as pd
import mne, scipy

from tsbench.data_pipeline.reader.base_reader import BaseReader

class EyeLinkReader(BaseReader):
    module_id = "eyelink"
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def fetch(self):
        df = self._asv_to_dataframe(self.data_path)
        # Remove potentially unused dimensions
        timeseries = [df[i].squeeze().astype(np.float32) for i in range(len(df))]
        return timeseries
    
    # Possible channels: [0, 1, 2]
    def _asv_to_dataframe(self, data_path, incl_channels = [2]):
        # Reading raw data
        raw = mne.io.read_raw_eyelink(data_path, verbose = "error",
                                    create_annotations=["messages"])
        
        # Finding applicable channels
        picks = [0, 1, 2]       # Specific to this dataset
        # Remove unwanted channels beforehand to speed up computations
        remv_picks = np.setdiff1d(picks, incl_channels)
        ch_names = raw.info.ch_names
        raw.drop_channels([ch_names[i] for i in remv_picks])

        # Find events
        events = mne.find_events(raw, "DIN", shortest_event=1, 
                                 min_duration=0.02, uint_cast=True, 
                                 verbose = "error")
    
        # Split data into separate event intervals
        event_id = {"flash": 3}

        epochs = mne.Epochs(
            raw, events, event_id, -0.3, 5, verbose="error",
            picks=[0], proj=False, decim=30, preload=True,
        )
        # Remove malformed data
        epochs.drop([17, 19], verbose = "error")
        
        # Return data
        return epochs._data
