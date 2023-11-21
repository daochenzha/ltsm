
class DataSplitter:
    def __init__(self):
        pass

    def get_splits(self):
        pass


class SplitterByTimestamp(DataSplitter):
    def __init__(self, seq_len, pred_len, train_ratio, val_ratio):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def get_splits(self, raw_data):
        train_split, val_split, test_split = [], [], []
        for sequence in raw_data:
            assert sequence.ndim == 1, "Time-series should be 1D."

            num_train = int(len(sequence) * self.train_ratio)
            num_val = int(len(sequence) * self.val_ratio)

            assert num_train >= self.seq_len + self.pred_len, f"Training sequence must have a lenth with at least seq_len + pred_len, the current length is {num_train}"

            # We also add the previous seq_len points to the val and test sets
            train_split.append(sequence[:num_train])
            val_split.append(sequence[num_train-self.seq_len:num_train+num_val])
            test_split.append(sequence[num_train+num_val-self.seq_len:])

            """
            print(num_train, num_val, len(sequence)-num_train-num_val)
            print(train_split[0].shape)
            print(val_split[0].shape)
            print(test_split[0].shape)
            print(len(sequence))
            """

        return train_split, val_split, test_split


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
