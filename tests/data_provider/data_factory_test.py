import pytest
import numpy as np
import pandas as pd
import os
import torch

from ltsm.data_provider.data_factory import _get_csv_prompt

@pytest.fixture
def setup_csv_file(tmpdir):
    print("Here")
    data = pd.DataFrame([range(133)])
    file_folder1 = tmpdir.mkdir("data_folder")
    file_folder = file_folder1.mkdir("data")
    file_path = os.path.join(file_folder, 'data_index_prompt.csv')
    data.to_csv(file_path, index=False)
    return file_folder1, file_path

@pytest.fixture
def setup_pth_file(tmpdir):
    data = pd.DataFrame([range(133)])
    assert data.shape == (1, 133)
    file_folder1 = tmpdir.mkdir("data_folder")
    file_folder = file_folder1.mkdir("data")
    file_path = os.path.join(file_folder, 'data_index_prompt.pth.tar')
    torch.save(data, file_path)
    return file_folder1, file_path

@pytest.fixture
def setup_npz_file(tmpdir):
    data = pd.DataFrame([range(133)])
    file_folder1 = tmpdir.mkdir("data_folder")
    file_folder = file_folder1.mkdir("data")
    file_path = os.path.join(file_folder, 'data_index_prompt.npz')
    np.savez(file_path, data=data.values, index=data.index, columns=data.columns)
    return file_folder1, file_path

def test_get_csv_prompt_csv(setup_csv_file):
    folder_path, file_path = setup_csv_file
    prompt_data_output = _get_csv_prompt(folder_path, 'dataset/data/data.csv', 'index')
    assert len(prompt_data_output) == 133

    arr = np.random.rand(366)
    concatenated_result = np.concatenate((prompt_data_output, arr))
    assert concatenated_result.shape == (133 + 366,)

def test_get_csv_prompt_pth(setup_pth_file):
    folder_path, file_path = setup_pth_file
    prompt_data_output = _get_csv_prompt(folder_path, 'dataset/data/data.csv', 'index')
    assert len(prompt_data_output) == 133

    arr = np.random.rand(366)
    concatenated_result = np.concatenate((prompt_data_output, arr))
    assert concatenated_result.shape == (133 + 366,)

def test_get_csv_prompt_npz(setup_npz_file):
    folder_path, file_path = setup_npz_file
    prompt_data_output = _get_csv_prompt(folder_path, 'dataset/data/data.csv', 'index')
    assert len(prompt_data_output) == 133

    arr = np.random.rand(366)
    concatenated_result = np.concatenate((prompt_data_output, arr))
    assert concatenated_result.shape == (133 + 366,)

