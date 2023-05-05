# encoding: utf-8
"""
@author:  wanglixiang
@contact: lixiangwang9705@gmail.com
"""


import numpy as np
import pandas as pd
import time
import os


def reduce_data(source_dir, target_dir, use_HDF5=False, use_feather=True):
    df = pd.read_csv(source_dir, parse_dates=True, keep_date_col=True)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:

            df[col] = df[col].astype('category')

    os.makedirs('/'.join(target_dir.split('/')[:-1]), exist_ok=True)

    if use_HDF5 == True and use_feather == False:
        data_store = pd.HDFStore(target_dir.replace('.csv', '.h5'))
        # Store object in HDFStore
        data_store.put('preprocessed_df', df, format='table')

        data_store.close()
    elif use_HDF5 == False and use_feather == True:
        df.to_feather(target_dir.replace('.csv', '.feather'))
    else:
        print('Please choose the only way to compress: True or False')

def reload_data(data_dir, use_HDF5=False, use_feather=True):
    if use_HDF5 == True and use_feather == False:
        store_data = pd.HDFStore(data_dir)
        preprocessed_df = store_data['preprocessed_df']
        store_data.close()

    elif use_HDF5 == False and use_feather == True:
        preprocessed_df = pd.read_feather(data_dir)

    else:
        print('Please choose the only way to compress: True or False')

    return preprocessed_df