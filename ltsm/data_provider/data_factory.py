from ltsm.data_provider.data_loader import (
    Dataset_Custom,
    Dataset_Pred,
    Dataset_TSF,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom_List,
    Dataset_Custom_List_TS
)

import os
import numpy as np
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'custom_list': Dataset_Custom_List,
    'custom_list_time_stamp': Dataset_Custom_List_TS,
}

def data_paths(dataset):
    '''
    args:
        dataset: string
            eeg_all, ecg_all, ecg_small_all,
            eeg_train, eeg_test, eeg_val, 
            ecg_train, ecg_test, ecg_val, 
            ecg_small_train, ecg_small_test, ecg_small_val
    return:
        data paths: list of strings
    '''
    paths = []
    if 'eeg' in dataset:
        for root, ds, fs in os.walk('/home/jy101/ltsm/dataset/eeg_csv/'):
            for f in fs:
                fullname = os.path.join(root, f)
                if fullname.endswith('.csv'):
                    paths.append(fullname)
    elif 'ecg' in dataset:
        for root, ds, fs in os.walk('/home/jy101/ltsm/dataset/fecgsyndb_csv/'):
            for f in fs:
                fullname = os.path.join(root, f)
                if fullname.endswith('.csv'):
                    paths.append(fullname)
    elif 'ecg_small' in dataset:
        for root, ds, fs in os.walk('/home/jy101/ltsm/dataset/ecg_arrhythmia_csv/'):
            for f in fs:
                fullname = os.path.join(root, f)
                if fullname.endswith('.csv'):
                    paths.append(fullname)
    else:
        pass
    
    length = len(paths)
    # random shuffle
    np.random.seed(0)
    np.random.shuffle(paths)

    split = dataset.split('_')[-1]
    assert split in ['all', 'train', 'test', 'val']
    type_map = {'all': 0, 'train': 1, 'val': 2, 'test': 3}
    set_type = type_map[split]
    
    num_train = int(len(length) * 0.7)
    num_test = int(len(length) * 0.2)
    num_vali = len(length) - num_train - num_test
    border1s = [0, 0, num_train, num_train + num_vali]
    border2s = [length, num_train, num_train + num_vali, len(length)]

    paths = paths[border1s[set_type]:border2s[set_type]]
    return paths
    

def get_data_loader(config, split, drop_last_test=True, train_all=False):
    Data = data_dict[config.data]
    timeenc = 0 if config.embed != 'timeF' else 1
    percent = config.percent
    max_len = config.max_len

    if split == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = config.batch_size
        freq = config.freq
    elif split == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = config.freq
        Data = Dataset_Pred
    elif split == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = config.batch_size
        freq = config.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = config.batch_size
        freq = config.freq

    if 'custom_list' in config.data:
        config.data_path = data_paths(config.data_path)
    
    data_set = Data(
        data_path=config.data_path,
        split=split,
        size=[config.seq_len, config.pred_len],
        features=config.features,
        target=config.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all
    )
    print(split, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=config.num_workers,
        drop_last=drop_last)

    return data_loader
