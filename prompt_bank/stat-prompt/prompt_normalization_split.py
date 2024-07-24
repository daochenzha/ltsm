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


def get_args():
    parser = argparse.ArgumentParser(description='LTSM')
    parser.add_argument('--mode', choices=["fit", "transform"], required=True)
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


def create_data_dir(dir_name):
    # prompt_dir =
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

if __name__ == "__main__":

    root_path_train = "./prompt_bank/stat-prompt/prompt_data_split/train"
    output_path_train = "./prompt_bank/stat-prompt/prompt_data_normalize_split/train"
    root_path_val = "./prompt_bank/stat-prompt/prompt_data_split/val"
    output_path_val = "./prompt_bank/stat-prompt/prompt_data_normalize_split/val"
    root_path_test = "./prompt_bank/stat-prompt/prompt_data_split/test"
    output_path_test = "./prompt_bank/stat-prompt/prompt_data_normalize_split/test"
    # normalize_param_fname = os.path.join(output_path, "normalization_params.pth.tar")
    ds_size = 50
    mode = get_args().mode # "transform" # "fit" #

    data_path_buf = {
        "train": {"root_path": root_path_train, "output_path": output_path_train, "normalize_param_fname": os.path.join(output_path_train, "normalization_params.pth.tar")},
        "val": {"root_path": root_path_val, "output_path": output_path_val, "normalize_param_fname": os.path.join(output_path_val, "normalization_params.pth.tar")},
        "test": {"root_path": root_path_test, "output_path": output_path_test, "normalize_param_fname": os.path.join(output_path_test, "normalization_params.pth.tar")},
    }


    dataset_name = [
        "electricity",
        "ETT-small",
        "exchange_rate",
        "illness",
        "traffic",
        "weather",
    ]

    for split_name, data_path in data_path_buf.items():

        root_path = data_path_buf[split_name]["root_path"]
        output_path = data_path_buf[split_name]["output_path"]
        normalize_param_fname = data_path_buf[split_name]["normalize_param_fname"]

        create_data_dir(output_path)

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










