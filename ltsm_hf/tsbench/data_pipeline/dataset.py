import numpy as np
import pandas as pd
import torch
import ipdb

from torch.utils.data.dataset import Dataset


class TSDataset(Dataset):
    def __init__(
        self, 
        data, 
        seq_len,
        pred_len,
    ):
        self.data = data
        self.seq_len = seq_len 
        self.pred_len = pred_len

        # Create a map from item index to sequence index and offset
        self.num_items = 0
        self.item2sequence, self.item2offset = [], []
        for sequence_index, sequence in enumerate(self.data):
            assert len(sequence) >= self.seq_len + self.pred_len, f"Sequence must have a lenth with at least seq_len + pred_len, the current length is {len(sequence)}"
            cur_offset = 0
            for _ in range(len(sequence) - self.seq_len - self.pred_len + 1):
                self.item2sequence.append(sequence_index)
                self.item2offset.append(cur_offset)
                cur_offset += 1
                self.num_items += 1

    def __getitem__(self, index):
        sequence_index = self.item2sequence[index]
        # TODO: Add label len
        x_begin = self.item2offset[index]
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len 
        
        seq_x = torch.from_numpy(np.expand_dims(self.data[sequence_index][x_begin:x_end], -1))
        seq_y = torch.from_numpy(np.expand_dims(self.data[sequence_index][y_begin:y_end], -1))

        return seq_x, seq_y

    def __len__(self):
        return self.num_items
    

class TSPromptDataset(Dataset):
    def __init__(
        self, 
        data, 
        prompt,
        seq_len,
        pred_len,
        downsample_rate=10,
        uniform_sampling=True,
    ):
        self.prompt = prompt
        self.seq_len = seq_len 
        self.pred_len = pred_len

        # Create a map from item index to sequence index and offset
        self.num_items = 0
        self.item2sequence, self.item2offset = [], []
        
        self.data = []
        for d in data:
            self.data.extend(d)
            
        if not uniform_sampling:
            # every dataset share same downsample rate
            for sequence_index, sequence in enumerate(self.data):
                assert len(sequence) >= self.seq_len + self.pred_len, f"Sequence must have a length with at least seq_len + pred_len, the current length is {len(sequence)}"
                cur_offset = 0
                for cur_offset in range(0, len(sequence) - self.seq_len - self.pred_len + 1, downsample_rate):
                    self.item2sequence.append(sequence_index)
                    self.item2offset.append(cur_offset)
                    # cur_offset += 1
                    self.num_items += 1
        else:
            # uniform sampling
            length = [] # record number of sequences of each dataset
            total_length = 0 # total number of sequences of all dataset
            # downsample rate is proportional to this number (downsample_rate*length[i]*number of datasets/total_length)
            for d in data:
                l = 0
                for i in range(len(d)):
                    assert len(d[i]) >= self.seq_len + self.pred_len, f"Sequence must have a length with at least seq_len + pred_len, the current length is {len(sequence)}"
                    l += len(d[i]) - self.seq_len - self.pred_len + 1
                length.append(l)
                total_length += l
            
            
            sequence_index = 0
            test_data = [0]*8
            for i in range(len(data)):
                downsample = int(len(length)*length[i]*downsample_rate/total_length)
                if downsample == 0:
                    downsample = 1
                for sequence in data[i]:
                    assert len(sequence) >= self.seq_len + self.pred_len, f"Sequence must have a length with at least seq_len + pred_len, the current length is {len(sequence)}"
                    for cur_offset in range(0, len(sequence) - self.seq_len - self.pred_len + 1, downsample):
                        self.item2sequence.append(sequence_index)
                        self.item2offset.append(cur_offset)
                        # cur_offset += 1
                        self.num_items += 1
                        test_data[i] += 1
                    sequence_index += 1
                    
            # ipdb.set_trace()
            
            

    def __getitem__(self, index):
        sequence_index = self.item2sequence[index]
        # TODO: Add label len
        x_begin = self.item2offset[index]
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len
        prompt_index = self.prompt[sequence_index]
        
        seq_x = np.concatenate((prompt_index, self.data[sequence_index][x_begin:x_end]))
        # seq_x = self.data[sequence_index][x_begin:x_end]
        seq_x = torch.from_numpy(np.expand_dims(seq_x, -1))
        seq_y = torch.from_numpy(np.expand_dims(self.data[sequence_index][y_begin:y_end], -1))
        # ipdb.set_trace()
        

        return seq_x, seq_y

    def __len__(self):
        return self.num_items


class TSPromptDatasetTXT(Dataset):
    def __init__(
        self, 
        data, 
        prompt,
        seq_len,
        pred_len,
        downsample_rate=10,
        uniform_sampling=True,
    ):
        self.prompt = prompt
        self.seq_len = seq_len 
        self.pred_len = pred_len

        # Create a map from item index to sequence index and offset
        self.num_items = 0
        self.item2sequence, self.item2offset = [], []
        
        self.data = []
        for d in data:
            self.data.extend(d)
            
        if not uniform_sampling:
            # every dataset share same downsample rate
            for sequence_index, sequence in enumerate(self.data):
                assert len(sequence) >= self.seq_len + self.pred_len, f"Sequence must have a length with at least seq_len + pred_len, the current length is {len(sequence)}"
                cur_offset = 0
                for cur_offset in range(0, len(sequence) - self.seq_len - self.pred_len + 1, downsample_rate):
                    self.item2sequence.append(sequence_index)
                    self.item2offset.append(cur_offset)
                    # cur_offset += 1
                    self.num_items += 1
        else:
            # uniform sampling
            length = [] # record number of sequences of each dataset
            total_length = 0 # total number of sequences of all dataset
            # downsample rate is proportional to this number (downsample_rate*length[i]*number of datasets/total_length)
            for d in data:
                l = 0
                for i in range(len(d)):
                    assert len(d[i]) >= self.seq_len + self.pred_len, f"Sequence must have a length with at least seq_len + pred_len, the current length is {len(sequence)}"
                    l += len(d[i]) - self.seq_len - self.pred_len + 1
                length.append(l)
                total_length += l
            
            
            sequence_index = 0
            test_data = [0]*8
            for i in range(len(data)):
                downsample = int(len(length)*length[i]*downsample_rate/total_length)
                if downsample == 0:
                    downsample = 1
                for sequence in data[i]:
                    assert len(sequence) >= self.seq_len + self.pred_len, f"Sequence must have a length with at least seq_len + pred_len, the current length is {len(sequence)}"
                    for cur_offset in range(0, len(sequence) - self.seq_len - self.pred_len + 1, downsample):
                        self.item2sequence.append(sequence_index)
                        self.item2offset.append(cur_offset)
                        # cur_offset += 1
                        self.num_items += 1
                        test_data[i] += 1
                    sequence_index += 1
                    
            # ipdb.set_trace()
            
            

    def __getitem__(self, index):
        sequence_index = self.item2sequence[index]
        # TODO: Add label len
        x_begin = self.item2offset[index]
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len
        prompt_index = self.prompt[sequence_index]
        # ipdb.set_trace()
        
        
        # seq_x = np.concatenate((prompt_index, self.data[sequence_index][x_begin:x_end]))
        # seq_x = self.data[sequence_index][x_begin:x_end]
        seq_x = torch.from_numpy(np.expand_dims(seq_x, -1))
        seq_y = torch.from_numpy(np.expand_dims(self.data[sequence_index][y_begin:y_end], -1))

        return seq_x, seq_y

    def __len__(self):
        return self.num_items
