import os
class DataSplitter:
    def __init__(self):
        pass

    def get_splits(self):
        pass


class SplitterByTimestamp(DataSplitter):
    def __init__(self, seq_len, pred_len, train_ratio, val_ratio,prompt_folder_path, data_name):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.prompt_folder_path = prompt_folder_path
        self.data_name = data_name
    
    def get_splits(self, raw_data):
        train_split, val_split, test_split, buff = [], [], [], []
        for index, sequence in enumerate(raw_data):
            # # if prompt_path exists, then we use this data
            # prompt_name = self.data_name.split("/")[-1]
            # prompt_name = prompt_name.replace(".tsf", "")
            # prompt_path = os.path.join(self.prompt_folder_path, prompt_name, "T"+str(index+1)+"_prompt.pth.tar")
            # if not os.path.exists(prompt_path):
            #     continue
            
            assert sequence.ndim == 1, "Time-series should be 1D."

            num_train = int(len(sequence) * self.train_ratio)
            num_val = int(len(sequence) * self.val_ratio)

            # assert num_train >= self.seq_len + self.pred_len, f"Training sequence must have a lenth with at least seq_len + pred_len, the current length is {num_train}"
            
            if num_train < self.seq_len + self.pred_len:
                continue
            
            
            # We also add the previous seq_len points to the val and test sets
            train_split.append(sequence[:num_train])
            val_split.append(sequence[num_train-self.seq_len:num_train+num_val])
            test_split.append(sequence[num_train+num_val-self.seq_len:])
            buff.append(index)
            

            """
            print(num_train, num_val, len(sequence)-num_train-num_val)
            print(train_split[0].shape)
            print(val_split[0].shape)
            print(test_split[0].shape)
            print(len(sequence))
            """

        return train_split, val_split, test_split, buff

    def get_csv_splits(self, df_data):
        train_split, val_split, test_split, buff = [], [], [], []
        cols = df_data.columns[1:]
        raw_data = df_data[cols].T.values
        if 'ETTh1' in self.data_name or 'ETTh2' in self.data_name:
            raw_data = df_data[cols][:14400].T.values

        if 'ETTm1' in self.data_name or 'ETTm2' in self.data_name:
            raw_data = df_data[cols][:57600].T.values

        for col, sequence in zip(cols, raw_data):
            # # if prompt_path exists, then we use this data
            # prompt_name = self.data_name.split("/")[-1]
            # prompt_name = prompt_name.replace(".tsf", "")
            # prompt_path = os.path.join(self.prompt_folder_path, prompt_name, "T"+str(index+1)+"_prompt.pth.tar")
            # if not os.path.exists(prompt_path):
            #     continue
            
            
            # index: the number of sequences. buff append the name of index. index is the column of the raw_data.
            
            assert sequence.ndim == 1, "Time-series should be 1D."

            num_train = int(len(sequence) * self.train_ratio)
            num_val = int(len(sequence) * self.val_ratio)

            # assert num_train >= self.seq_len + self.pred_len, f"Training sequence must have a lenth with at least seq_len + pred_len, the current length is {num_train}"
            
            if num_train < self.seq_len + self.pred_len:
                continue
            
            
            # We also add the previous seq_len points to the val and test sets
            train_split.append(sequence[:num_train])
            val_split.append(sequence[num_train-self.seq_len:num_train+num_val])
            test_split.append(sequence[num_train+num_val-self.seq_len:])
            buff.append(col)
            

            """
            print(num_train, num_val, len(sequence)-num_train-num_val)
            print(train_split[0].shape)
            print(val_split[0].shape)
            print(test_split[0].shape)
            print(len(sequence))
            """
        print(f"Data{self.data_name} has been split into train, val, test sets with the following shapes: {train_split[0].shape}, {val_split[0].shape}, {test_split[0].shape}")

        return train_split, val_split, test_split, buff


class SplitterByTimeseries(DataSplitter):
    def __init__(self, config):
        super(SplitByTimeseries, self).__init__()

    def split(self):
        pass

    def train_data(self):
        pass

    def val_data(self):
        pass

    def test_data(self):
        pass
