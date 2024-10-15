from ltsm.common.base_splitter import DataSplitter
import pandas as pd
import numpy as np

from typing import Tuple, List

class SplitterByTimestamp(DataSplitter):
    def __init__(self, seq_len, pred_len, train_ratio, val_ratio,prompt_folder_path):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.prompt_folder_path = prompt_folder_path
    
    def get_splits(self, raw_data):
        train_split, val_split, test_split, buff = [], [], [], []
        for index, sequence in enumerate(raw_data):
            
            assert sequence.ndim == 1, "Time-series should be 1D."

            num_train = int(len(sequence) * self.train_ratio)
            num_val = int(len(sequence) * self.val_ratio)
                        
            if num_train < self.seq_len + self.pred_len:
                continue
                 
            # We also add the previous seq_len points to the val and test sets
            train_split.append(sequence[:num_train])
            val_split.append(sequence[num_train-self.seq_len:num_train+num_val])
            test_split.append(sequence[num_train+num_val-self.seq_len:])
            buff.append(index)

        return train_split, val_split, test_split, buff

    def get_csv_splits(self, df_data: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Splits the data into training-validation-training sets.

        Args:
            df_data (pd.DataFrame): A Pandas DataFrame containing the data to be split.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                A tuple containing fours lists of sequences for the training, validation, and test sets. 
                The last list contains the row labels of these sequences.
        """
        train_split, val_split, test_split, buff = [], [], [], []
        raw_data = df_data.to_numpy()

        for index, sequence in zip(df_data.index, raw_data):
            
            assert sequence.ndim == 1, "Time-series should be 1D."

            num_train = int(len(sequence) * self.train_ratio)
            num_val = int(len(sequence) * self.val_ratio)
            
            if num_train < self.seq_len + self.pred_len:
                continue
            
            
            # We also add the previous seq_len points to the val and test sets
            train_split.append(sequence[:num_train])
            val_split.append(sequence[num_train-self.seq_len:num_train+num_val])
            test_split.append(sequence[num_train+num_val-self.seq_len:])
            buff.append(index)

        return train_split, val_split, test_split, buff
