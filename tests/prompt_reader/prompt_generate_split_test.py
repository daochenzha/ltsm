import os
import pytest
import pandas as pd
import numpy as np
import torch

@pytest.fixture
def setup(mocker, tmp_path):
    """set up the test environment"""
    mocker.patch.dict('sys.modules', {'tsfel': mocker.MagicMock()})
    from ltsm.prompt_reader.stat_prompt.prompt_generate_split import prompt_save

    sample_prompt_buf = {
        'train': pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10)
        }),
        'val': pd.DataFrame({
            'feature1': np.random.rand(5),
            'feature2': np.random.rand(5)
        }),
        'test': pd.DataFrame({
            'feature1': np.random.rand(5),
            'feature2': np.random.rand(5)
        })
    }

    output_path = str(tmp_path)
    data_name = "test_data"
    ifTest = False  

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_path, split)
        os.makedirs(split_dir, exist_ok=True)
    
    return prompt_save, sample_prompt_buf, output_path, data_name,  ifTest

@pytest.mark.parametrize("save_format", ["pth.tar", "csv", "npz"])
def test_prompt_save(setup, save_format):
    """test if the prompt data is saved correctly in different formats and loaded back correctly
    """
    prompt_save, sample_prompt_buf, output_path, data_name, ifTest = setup
    prompt_save(sample_prompt_buf, output_path, data_name, save_format, ifTest)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_path, split)
        for index, col in sample_prompt_buf[split].T.iterrows():
            file_name = f"{data_name}_{index}_prompt.{save_format}"
            file_path = os.path.join(split_dir, file_name)
            assert os.path.exists(file_path), f"File {file_path} does not exist"

            prompt_data = col
            prompt_data.columns = [index]
            prompt_data = prompt_data.T

            if save_format == "pth.tar":
                load_data = torch.load(file_path)
            elif save_format == "csv":
                load_data = pd.read_csv(file_path)
                if isinstance(load_data, pd.DataFrame):
                    load_data = load_data.squeeze()
            elif save_format == "npz":
                loaded = np.load(file_path)
                load_data = pd.Series(data=loaded["data"], index=loaded["index"], name=loaded["name"].item())
                if isinstance(load_data, pd.DataFrame):
                    load_data = load_data.squeeze()
            else:
                raise ValueError(f"Unsupported save format: {save_format}")

            assert type(load_data) == type(prompt_data), f"Type mismatch: {type(load_data)} vs {type(prompt_data)}"
            assert load_data.shape == prompt_data.shape, f"Shape mismatch: {load_data.shape} vs {prompt_data.shape}"
            assert load_data.index.equals(prompt_data.index), "Index mismatch"
            assert load_data.name == prompt_data.name, f"Series names mismatch: {load_data.name} vs {prompt_data.name}"
            assert np.allclose(load_data.values, prompt_data.values, rtol=1e-8, atol=1e-8), "Data values mismatch"
            if save_format != "csv":
                assert load_data.equals(prompt_data), f"Data mismatch: {load_data} vs {prompt_data}"
            print(f"All tests passed for {file_path}")
