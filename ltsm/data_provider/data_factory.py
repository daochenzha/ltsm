from ltsm.data_provider.data_loader import (
    Dataset_Custom,
    Dataset_Pred,
    Dataset_TSF,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom_List,
    Dataset_Custom_List_TS,
    Dataset_Custom_List_TS_TSF
)

import os
import numpy as np
from torch.utils.data import DataLoader
import copy
import pandas as pd

data_dict = {
    'custom': Dataset_Custom,
    'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'custom_list': Dataset_Custom_List,
    'custom_list_time_stamp': Dataset_Custom_List_TS,
    'tsf_list': Dataset_Custom_List_TS_TSF,
}

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

    data_set = Data(
        data_path=config.data_path,
        split=split,
        size=[config.seq_len, config.label_len, config.pred_len],
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


def data_paths(data_path):
    '''
    args:
        data_path: string
    return:
        data paths: list of strings
    '''
    paths = []
    root_path = f'/home/jy101/ltsm/dataset/{data_path}/' if '/home/' not in data_path else data_path
    for root, ds, fs in os.walk(root_path):
        for f in fs:
            fullname = os.path.join(root, f)
            if fullname.endswith('.csv') or fullname.endswith('.feather'):
                paths.append(fullname)
    return paths


def get_data_loaders(config, drop_last_test=True, train_all=False):
    Data = data_dict[config.data]
    timeenc = 0 if config.embed != 'timeF' else 1
    percent = config.percent
    max_len = config.max_len
    freq = config.freq
    batch_size=config.batch_size

    data_path_all = data_paths(config.data_path)
    length = len(data_path_all)

    np.random.seed(0)
    np.random.shuffle(data_path_all)

    num_train = int(length * 0.7)
    num_test = int(length * 0.2)
    num_vali = length - num_train - num_test
    border1s = [0, num_train, num_train + num_vali]
    border2s = [num_train, num_train + num_vali, length]

    data_path_train = data_path_all[border1s[0]:border2s[0]]
    data_path_val = data_path_all[border1s[1]:border2s[1]]
    data_path_test = data_path_all[border1s[2]:border2s[2]]
    
    if 'time_stamp' in config.data:
        data_path_train = []
        data_path_val = []
        data_path_test = []
    
    train_set = Data(data_path=data_path_train,
        split='train',
        size=[config.seq_len, config.pred_len],
        features=config.features,
        target=config.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all)
    test_set = Data(data_path=data_path_val,
        split='test',
        size=[config.seq_len, config.pred_len],
        features=config.features,
        target=config.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all)
    val_set = Data(data_path=data_path_test,
        split='val',
        size=[config.seq_len, config.pred_len],
        features=config.features,
        target=config.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all)

    if 'time_stamp' in config.data:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        for path in data_path_all:
            if path.endswith('.csv'):
                df_raw = pd.read_csv(path)
            elif path.endswith('.feather'):
                df_raw = pd.read_feather(path)
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - config.seq_len, len(df_raw) - num_test - config.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            
            border2s[0] = (border2s[0] - config.seq_len) * percent // 100 + config.seq_len

            # scaling
            train_data = df_raw[border1s[0]:border2s[0]]
            scaler.fit(train_data.values)
            data = scaler.transform(df_raw.values)

            train_set.add_data(data[border1s[0]:border2s[0]])
            val_set.add_data(data[border1s[1]:border2s[1]])
            test_set.add_data(data[border1s[2]:border2s[2]])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True)
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=drop_last_test)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=drop_last_test)
    
    return train_loader, val_loader, test_loader