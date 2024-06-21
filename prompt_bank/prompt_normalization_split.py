# from ltsm.data_provider.data_factory import get_data_loader, get_data_loaders, get_dataset
import argparse
import ipdb
import pandas as pd
import numpy as np
# import tsfel
from pandas import read_csv, read_feather
import matplotlib.pyplot as plt
import sys, os
import torch
from sklearn.preprocessing import StandardScaler

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


def prompt_generation(ts):
    cfg = tsfel.get_features_by_domain()
    prompt = tsfel.time_series_features_extractor(cfg, ts)
    return prompt


def prompt_prune(pt):
    pt_dict = pt.to_dict()
    pt_keys = list(pt_dict.keys())
    for key in pt_keys:
        if type(key) == type("abc") and key.startswith("0_FFT mean coefficient"):
            del pt[key]

    return pt


def mean_std_export_ds(data_path_buf, normalize_param_fname):
    prompt_data_buf = []
    output_dir_buf = []
    output_path_buf = []
    for index, dataset_path in enumerate(data_path_buf):
        prompt_data = torch.load(dataset_path)
        prompt_data = prompt_prune(prompt_data)
        # print(prompt_data)
        prompt_data_buf.append(prompt_data)

        data_name = dataset_path.replace(root_path, "").replace(".csv", "")
        data_dir = data_name[0:data_name.rfind("/")]
        prompt_dir = os.path.join(output_path, data_dir)
        prompt_fname = os.path.join(output_path, data_name)
        # print(prompt_fname)
        output_dir_buf.append(prompt_dir)
        output_path_buf.append(prompt_fname)
        print("Import from {}".format(dataset_path), prompt_data.shape)
        # ipdb.set_trace()

    prompt_data_all = pd.concat(prompt_data_buf, axis=1).T
    print(prompt_data_all)

    scaler = StandardScaler()
    scaler.fit(prompt_data_all)

    sc_mean = pd.DataFrame(scaler.mean_.reshape(1,-1), columns=prompt_data_all.keys())
    sc_scale = pd.DataFrame(scaler.scale_.reshape(1,-1), columns=prompt_data_all.keys())

    print({"mean": sc_mean, "scale": sc_scale})
    print("Save the mean and std to {}".format(normalize_param_fname))
    torch.save({"mean": sc_mean, "scale": sc_scale}, normalize_param_fname)


def standardscale_export(data_path_buf, params_fname, output_path, root_path):

    params = torch.load(params_fname)
    mean, std = params["mean"], params["scale"]
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = std
    # ipdb.set_trace()

    for index, dataset_path in enumerate(data_path_buf):
        prompt_data_raw = torch.load(dataset_path)
        prompt_data_raw = prompt_prune(prompt_data_raw)

        prompt_data = scaler.transform(prompt_data_raw.values.reshape(1, -1))
        prompt_data_array = prompt_data
        # print(prompt_data)
        prompt_data_array[np.isnan(prompt_data_array)] = 0
        prompt_data_transform = pd.DataFrame(prompt_data_array, columns=prompt_data.keys())
        # ipdb.set_trace()

        prompt_fname = dataset_path.replace(root_path, output_path)
        prompt_dir = prompt_fname[0:prompt_fname.rfind("/")]
        if not os.path.exists(prompt_dir):
            os.mkdir(prompt_dir)

        torch.save(prompt_data_transform, prompt_fname)
        print("Save to {}".format(prompt_fname))
        del prompt_data


if __name__ == "__main__":

    root_path = "./prompt_bank/prompt_data_split/train"
    output_path = "./prompt_bank/prompt_data_normalize_split/train"
    root_path = "./prompt_bank/prompt_data_split/val"
    output_path = "./prompt_bank/prompt_data_normalize_split/val"
    root_path = "./prompt_bank/prompt_data_split/test"
    output_path = "./prompt_bank/prompt_data_normalize_split/test"
    normalize_param_fname = os.path.join(output_path, "normalization_params.pth.tar")
    ds_size = 50
    mode = "transform" # "fit" #

    # data_path_buf = data_paths(root_path)
    # print(data_path_buf)

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
    if mode == "fit":

        for dataset_dir in dataset_fullname:
            paths = os.listdir(dataset_dir)
            new_dataset = [os.path.join(dataset_dir, path) for path in paths]
            sample_idx = np.random.permutation(len(new_dataset))[:ds_size].astype(np.int64)
            # ipdb.set_trace()
            new_dataset = np.array(new_dataset)[sample_idx].tolist()
            data_path_buf.extend(new_dataset)

    else:
        for dataset_dir in dataset_fullname:
            paths = os.listdir(dataset_dir)
            new_dataset = [os.path.join(dataset_dir, path) for path in paths]
            data_path_buf.extend(new_dataset)


    if mode == "fit":

        mean_std_export_ds(data_path_buf, normalize_param_fname)
    else:
        # ipdb.set_trace()
        standardscale_export(data_path_buf, normalize_param_fname, output_path, root_path)










