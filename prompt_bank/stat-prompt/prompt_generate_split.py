# from ltsm.data_provider.data_factory import get_data_loader, get_data_loaders, get_dataset
import argparse
import ipdb
import pandas as pd
import numpy as np
import tsfel
from pandas import read_csv, read_feather
import matplotlib.pyplot as plt
import sys, os
import torch

from ltsm.data_provider.data_factory import data_paths

def get_args():
    parser = argparse.ArgumentParser(description='LTSM')

    parser.add_argument('--data_path', type=str, default='dataset/weather.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--freq', type=str, default="h")
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--features', type=str, default='M')

    args = parser.parse_args()

    return args

def prompt_prune(pt):
    pt_dict = pt.to_dict()
    pt_keys = list(pt_dict.keys())
    for key in pt_keys:
        if key.startswith("0_FFT mean coefficient"):
            del pt[key]

    return pt


def prompt_generation_single(ts):
    cfg = tsfel.get_features_by_domain()
    prompt = tsfel.time_series_features_extractor(cfg, ts)
    prompt = prompt_prune(prompt)
    return prompt

def prompt_generation(ts, ts_name):

    print(ts.shape)

    if ts.shape[1] == 1:

        return None

    else:

        column_name = [name.replace("/", "-") for name in list(ts.columns)]
        prompt_buf_train = pd.DataFrame(np.zeros((133, ts.shape[1])), columns=column_name)
        prompt_buf_val = pd.DataFrame(np.zeros((133, ts.shape[1])), columns=column_name)
        prompt_buf_test = pd.DataFrame(np.zeros((133, ts.shape[1])), columns=column_name)
        for index, col in ts.T.iterrows():
            if "ETT" in ts_name:
                ts_len = len(ts)
                t1, t2 = int(0.6*ts_len), int(0.6*ts_len) + int(0.2*ts_len)
                ts_train, ts_val, ts_test = col[:t1], col[t1:t2].reset_index(drop=True), col[t2:].reset_index(drop=True)
            else:
                ts_len = len(ts)
                t1, t2 = int(0.7 * ts_len), int(0.7 * ts_len) + int(0.1 * ts_len)
                ts_train, ts_val, ts_test = col[:t1], col[t1:t2].reset_index(drop=True), col[t2:].reset_index(drop=True)

            prompt_train = prompt_generation_single(ts_train)
            prompt_val = prompt_generation_single(ts_val)
            prompt_test = prompt_generation_single(ts_test)

            prompt_buf_train[index.replace("/", "-")] = prompt_train.T.values
            prompt_buf_val[index.replace("/", "-")] = prompt_val.T.values
            prompt_buf_test[index.replace("/", "-")] = prompt_test.T.values

    prompt_buf_total = {"train": prompt_buf_train, "val": prompt_buf_val, "test": prompt_buf_test}
    print(prompt_buf_total)
    return prompt_buf_total


def prompt_save(prompt_buf, output_path):

    print(prompt_buf["train"])
    if prompt_buf["train"].shape[1] == 1:
        # ipdb.set_trace()
        return None

        # prompt_train_fname = os.path.join(prompt_train_data_dir, data_name + "_prompt.pth.tar")
        # prompt_train = prompt_buf["train"]
        # print("Export", prompt_train_fname, prompt_train.shape)
        #
        # prompt_val_fname = os.path.join(prompt_val_data_dir, data_name + "_prompt.pth.tar")
        # prompt_val = prompt_buf["val"]
        # torch.save(prompt_val, prompt_val_fname)
        # print("Export", prompt_val_fname, prompt_val.shape)
        #
        # prompt_test_fname = os.path.join(prompt_test_data_dir, data_name + "_prompt.pth.tar")
        # prompt_test = prompt_buf["test"]
        # torch.save(prompt_test, prompt_test_fname)
        # print("Export", prompt_test_fname, prompt_test.shape)

    else:

        for index, col in prompt_buf["train"].T.iterrows():

            prompt_train_fname = os.path.join(output_path, "train", data_name + "_" + index + "_prompt.pth.tar")
            prompt_train = col
            prompt_train.columns = [index]
            prompt_train = prompt_train.T
            torch.save(prompt_train, prompt_train_fname)
            print("Export", prompt_train_fname, prompt_train.shape)

        for index, col in prompt_buf["val"].T.iterrows():
            prompt_val_fname = os.path.join(output_path, "val", data_name + "_" + index + "_prompt.pth.tar")
            prompt_val = col
            prompt_val.columns = [index]
            prompt_val = prompt_val.T
            torch.save(prompt_val, prompt_val_fname)
            print("Export", prompt_val_fname, prompt_val.shape)

        for index, col in prompt_buf["test"].T.iterrows():
            prompt_test_fname = os.path.join(output_path, "test", data_name + "_" + index + "_prompt.pth.tar")
            prompt_test = col
            prompt_test.columns = [index]
            prompt_test = prompt_test.T
            torch.save(prompt_test, prompt_test_fname)
            print("Export", prompt_test_fname, prompt_test.shape)


def data_import(path, format="feather"):

    if format == "feather":
        data = read_feather(path)
        data_name = path.replace(root_path, "").replace(".feather", "")
        data_dir = data_name[0:data_name.rfind("/")]
        # ipdb.set_trace()
        data = data.value

    else:
        data = read_csv(path)
        data_name = path.replace(root_path, "").replace(".csv", "")
        data_dir = data_name[0:data_name.rfind("/")]
        if "date" in data.columns:
            data = data.drop("date", axis=1)
        # print(data)
        # data = data.value


    return data, data_name, data_dir


def create_data_dir(dir_name):
    # prompt_dir =
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":

    root_path = "../../datasets/"
    output_path = "./prompt_bank/stat-prompt/prompt_data_split/"


    dataset_name = [
        "electricity",
        "ETT-small",
        "exchange_rate",
        "illness",
        "traffic",
        "weather",
    ]

    dataset_fullname = [os.path.join(root_path, name) for name in dataset_name]
    data_path_buf = []
    for dataset_dir in dataset_fullname:
        paths = data_paths(dataset_dir)
        data_path_buf.extend(paths)

    print(data_path_buf)
    # ipdb.set_trace()

    for path_idx, path in enumerate(data_path_buf):

        # print(path)

        data, data_name, data_dir = data_import(path, "csv")
        # print("Data Shape:", data.shape)
        if data.shape[0] < 20:
            print(path, "Skip too short time-series data.", data.shape)
            continue
        else:
            print("Import", path, "data shape", data.shape)

        create_data_dir(os.path.join(output_path, "train"))
        create_data_dir(os.path.join(output_path, "val"))
        create_data_dir(os.path.join(output_path, "test"))
        create_data_dir(os.path.join(output_path, "train", data_dir))
        create_data_dir(os.path.join(output_path, "val", data_dir))
        create_data_dir(os.path.join(output_path, "test", data_dir))

        prompt_data_buf = prompt_generation(data, data_name)
        if prompt_data_buf is not None:
            prompt_save(prompt_data_buf, output_path)

