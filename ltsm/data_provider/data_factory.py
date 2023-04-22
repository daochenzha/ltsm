from ltsm.data_provider.data_loader import (
    Dataset_Custom,
    Dataset_Pred,
    Dataset_TSF,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom_List,
)

from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'custom_list': Dataset_Custom_List,
}

def data_paths(dataset):
    '''
    args:
        dataset: string
            eeg_train, eeg_test, eeg_val, 
            ecg_train, ecg_test, ecg_val, 
            ecg_small_train, ecg_small_test, ecg_small_val
    return:
        data paths: list of strings
    '''
    pass
    

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

    if config.data == 'custom_list':
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
