import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path

from torch.utils.data.dataset import ConcatDataset, Dataset

from ltsm.utils.timefeatures import time_features
from ltsm.utils.tools import convert_tsf_to_dataframe

warnings.filterwarnings('ignore')

class HF_Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        # for key, value in vars(dataset).items():
        #     setattr(self, key, value)

    def __read_data__(self):
        return self.dataset.__read_data__()

    def __len__(self):
        return self.dataset.__len__()

    def inverse_transform(self, data):
        return self.dataset.inverse_transform(data)

    def add_data(self, df):
        return self.dataset.add_data(df)

    def __getitem__(self, index):

        seq_x, seq_y = self.dataset.__getitem__(index)

        return {
            "input_data": seq_x,
            "labels": seq_y
        }
        
        # return {
        #     "input_data": seq_x,
        #     "labels": seq_y,
        #     "seq_x_mark": seq_x_mark,
        #     "seq_y_mark": seq_y_mark,
        # }
