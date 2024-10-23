import os
import pytest
import torch
import pandas as pd
import numpy as np

from ltsm.prompt_reader.stat_prompt.prompt_normalization_split import save_data 

@pytest.fixture
def setup():
    """input data for testing"""
    data = pd.DataFrame([range(133)])
    print(data.shape)
    return data

@pytest.mark.parametrize("save_format", ["pth.tar", "csv", "npz"])
def test_save_data(tmpdir, setup, save_format):
    """test save_data function: save data in different formats and load it back to check if the data is saved correctly"""
    data_path = os.path.join(tmpdir, f"test_data.{save_format}")
    
    save_data(setup, data_path, save_format)
    
    if save_format == "pth.tar":
        loaded_data = torch.load(data_path)
    elif save_format == "csv":
        loaded_data = pd.read_csv(data_path)
        loaded_data.columns = loaded_data.columns.astype(int)
    elif save_format == "npz":
        loaded = np.load(data_path)
        loaded_data = pd.DataFrame(data=loaded["data"])

    assert isinstance(loaded_data, pd.DataFrame), "Loaded data should be a DataFrame"
    assert loaded_data.shape == setup.shape, f"Shape mismatch: {loaded_data.shape} vs {setup.shape}"
    assert loaded_data.columns.equals(setup.columns), "Columns mismatch"
    assert np.allclose(loaded_data.values, setup.values, rtol=1e-8, atol=1e-8), "Data values mismatch"

