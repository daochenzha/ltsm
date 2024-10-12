import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ltsm.data_reader import reader_dict
from ltsm.data_provider.data_splitter import SplitterByTimestamp
from ltsm.data_provider.tokenizer import processor_dict
from ltsm.data_provider.dataset import TSDataset,  TSPromptDataset, TSTokenDataset

from typing import Tuple, List
from ltsm.common.base_processor import BaseProcessor
import logging
import ipdb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class DatasetFactory:
    def __init__(
        self, 
        data_paths: List[str], 
        prompt_data_path: str, 
        data_processing: str, 
        seq_len: int, 
        pred_len: int, 
        train_ratio: float, 
        val_ratio: float,
        model: str= None,
        scale_on_train: bool = False, 
        downsample_rate: int = 10
    ):
        self.data_paths = data_paths
        self.prompt_data_path = prompt_data_path
        self.model = model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale_on_train = scale_on_train
        self.downsample_rate = downsample_rate
        self.splitter = SplitterByTimestamp(
            seq_len,
            pred_len,
            train_ratio,
            val_ratio,
            prompt_folder_path=self.prompt_data_path
        )
        self.processor = processor_dict[data_processing]

    def fetch(self, data_path: str)->pd.DataFrame:
        dir_name = os.path.split(os.path.dirname(data_path))[-1]
        return reader_dict[dir_name](data_path).fetch()
    
    def splitData(self, df: pd.DataFrame)->Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        return self.splitter.get_csv_splits(df)
    
    def process(self, train: List[np.ndarray], val: List[np.ndarray], test: List[np.ndarray], data_name:str, data: np.ndarray)->Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        train, val, test = self.processor.process(
            data,
            train,
            val,
            test,
            fit_train_only=self.scale_on_train
        )
        logging.info(f"Data{data_name} has been split into train, val, test sets with the following shapes: {train[0].shape}, {val[0].shape}, {test[0].shape}")
        return train, val, test
    
    def __get_prompt(self, prompt_data_path:str, data_name: str, idx_file_name: str):
        dir_name = os.path.split(os.path.dirname(data_name))[-1]
        idx_file_name = idx_file_name.replace("/", "-")
        idx_file_name = idx_file_name.replace("**", "_")
        idx_file_name = idx_file_name.replace("%", "_")
        if dir_name == "csv":
            prompt_name = data_name.split('/')[-2]+'/'+data_name.split('/')[-1].split('.')[0]
            prompt_path = os.path.join(prompt_data_path,prompt_name+'_'+str(idx_file_name)+"_prompt.pth.tar")
        elif dir_name == "monash":
            prompt_name = data_name.split("/")[-1]
            prompt_name = prompt_name.replace(".tsf", "")
            prompt_path = os.path.join(prompt_data_path, prompt_name, "T"+str(idx_file_name+1)+"_prompt.pth.tar")
        
        if not os.path.exists(prompt_path):
            logging.error(f"Prompt file {prompt_path} does not exist")
            return
        
        prompt_data = torch.load(prompt_path)
        prompt_data = prompt_data.T[0]
        
        prompt_data = [ prompt_data.iloc[i] for i in range(len(prompt_data)) ]
        return prompt_data
    
    def loadPrompts(self, data_path: str, prompt_data_path:str, buff: List[np.array])->List[np.ndarray]:
        prompt_data = []
        if self.model == 'WordPrompt':
            for _ in buff:
                prompt_data.append([self.data_paths.index(data_path)])
        else:
            for instance_idx in buff:
                instance_prompt = self.__get_prompt(
                    prompt_data_path,  
                    data_path,
                    instance_idx
                )
                prompt_data.append(instance_prompt)
            return prompt_data
    
    def createTorchDS(self, data: List[np.ndarray], prompt_data: List[np.ndarray], downsample_rate: int)->TSDataset:
        if len(data) == 0 or len(prompt_data) == 0:
            return None
        
        if self.model == 'Tokenizer':
            return TSTokenDataset(
                data=data,
                prompt=prompt_data,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                downsample_rate=downsample_rate
            )
        else:
            return TSPromptDataset(
                data=data,
                prompt=prompt_data,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                downsample_rate=downsample_rate
            )

    def getDatasets(self)->Tuple[TSDataset, TSDataset, List[TSDataset], BaseProcessor]:
        train_data, val_data, train_prompt_data, val_prompt_data, test_ds_list = [], [], [], [], []
        for data_path in self.data_paths:
            # Step 0: Read data, the output is a list of 1-d time-series
            df_data = self.fetch(data_path)

            # Step 1: Get train, val, and test splits
            sub_train_data, sub_val_data, sub_test_data, buff = self.splitData(df_data)

            # Step 2: Scale the datasets. We fit on the whole sequence by default.
            # To fit on the train sequence only, set scale_on_train=True
            sub_train_data, sub_val_data, sub_test_data = self.process(sub_train_data, sub_val_data, sub_test_data, data_path, df_data.to_numpy())
            train_data.extend(sub_train_data)
            val_data.extend(sub_val_data)

            # Step 2.5: Load prompt for each instance
            # Train Prompt
            train_prompt_data_path = self.prompt_data_path + '/train'
            train_prompt_data.extend(self.loadPrompts(data_path, train_prompt_data_path, buff))
            
            # Validation Prompt
            val_prompt_data_path = self.prompt_data_path + '/val'
            val_prompt_data.extend(self.loadPrompts(data_path, val_prompt_data_path, buff))

            # Test Prompt
            test_prompt_data_path = self.prompt_data_path + '/test'
            test_prompt_data = self.loadPrompts(data_path, test_prompt_data_path, buff)

            # Create a Torch dataset for each sub test dataset
            test_ds_list.append(self.createTorchDS(sub_test_data, test_prompt_data, 1))
        
        # Step 3: Create Torch datasets (samplers)
        train_ds = self.createTorchDS(train_data, train_prompt_data, self.downsample_rate)
        dir_name = os.path.split(os.path.dirname(self.data_paths[0]))[-1]
        if dir_name == 'monash':
            val_ds = self.createTorchDS(val_data, val_prompt_data, 54)
        else:
            val_ds = self.createTorchDS(val_data, val_prompt_data, self.downsample_rate)
        
        return train_ds, val_ds, test_ds_list, self.processor


def create_csv_datasets(
    data_path,
    prompt_data_path,
    data_processing,
    seq_len,
    pred_len,
    train_ratio,
    val_ratio,
    scale_on_train=False,
    downsample_rate=10,
):
    # Load Training data and validation data
    train_data, val_data, test_data, train_prompt_data, val_prompt_data, test_prompt_data = [], [], [], [], [], []
    for sub_data_path in data_path:
        sub_train_ratio = train_ratio
        sub_val_ratio = val_ratio
        
        # Step 0: Read data, the output is a list of 1-d time-series
        df_data = pd.read_csv(sub_data_path)
        cols = df_data.columns[1:] 
        raw_data = df_data[cols].T.values
        
        # keep the ETT data setting same with PatchTST (dataset length and train/val/test ratio)
        if 'ETTh1' in sub_data_path or 'ETTh2' in sub_data_path:
            sub_train_ratio = 0.6
            sub_val_ratio = 0.2
        if 'ETTm1' in sub_data_path or 'ETTm2' in sub_data_path:
            sub_train_ratio = 0.6
            sub_val_ratio = 0.2
        test_ratio = 1.0 - sub_train_ratio - sub_val_ratio
        print(f"Training Loading {sub_data_path}, train {sub_train_ratio:.2f}, val {sub_val_ratio:.2f}, test {test_ratio:.2f}")


        # Step 1: Get train, val, and test splits
        # For now, we use SplitterByTimestamp only
        sub_train_data, sub_val_data, sub_test_data, buff = SplitterByTimestamp(
            seq_len,
            pred_len,
            train_ratio=sub_train_ratio,
            val_ratio=sub_val_ratio,
            prompt_folder_path=prompt_data_path,
            data_name=sub_data_path
        ).get_csv_splits(df_data)


        # Step 2: Scale the datasets. We fit on the whole sequence by default.
        # To fit on the train sequence only, set scale_on_train=True
        # For now, we use StandardScaler only
        processor = processor_dict[data_processing]() 
        sub_train_data, sub_val_data, sub_test_data = processor.process(
            raw_data,  # Used for scaling
            sub_train_data,
            sub_val_data,
            sub_test_data,
            fit_train_only=scale_on_train,
        )

        # Step 2.5 Load prompt for each instance
        # Train Prompt
        train_prompt_data_path = prompt_data_path + '/train'
        for train_instance_idx in buff:
            instance_prompt =_get_csv_prompt(
                train_prompt_data_path,  
                sub_data_path,
                train_instance_idx
            )
            train_prompt_data.append(instance_prompt)
        
        val_prompt_data_path = prompt_data_path + '/val'
        for val_instance_idx in buff:
            instance_prompt = _get_csv_prompt(
                val_prompt_data_path,  
                sub_data_path,
                val_instance_idx
            )
            val_prompt_data.append(instance_prompt)
        
        train_data.extend(sub_train_data)
        val_data.extend(sub_val_data)


    # Step 3: Create Torch datasets (samplers)
    train_dataset = TSPromptDataset (
        data=train_data,
        prompt=train_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
        uniform_sampling=False
    )

    val_dataset = TSPromptDataset (
        data=val_data,
        prompt=val_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
        uniform_sampling=False
    )

    return train_dataset, val_dataset, processor

def create_csv_test_datasets(
    test_data_path,
    prompt_data_path,
    data_processing,
    seq_len,
    pred_len,
    train_ratio,
    val_ratio,
    scale_on_train=False,
    downsample_rate=10,
):
    # Load Testing data
    test_train_ratio, test_val_ratio = train_ratio, val_ratio

    if 'ETTh1' in test_data_path or 'ETTh2' in test_data_path:
        test_train_ratio = 0.6
        test_val_ratio = 0.2
    if 'ETTm1' in test_data_path or 'ETTm2' in test_data_path:
        test_train_ratio = 0.6
        test_val_ratio = 0.2
    test_test_ratio = 1.0 - test_train_ratio - test_val_ratio
    print(f"Testing Loading {test_data_path}, train {test_train_ratio:.2f}, val {test_val_ratio:.2f}, test {test_test_ratio:.2f}")

    # Step 0: Read data, the output is a list of 1-d time-series
    df_data = pd.read_csv(test_data_path)
    cols = df_data.columns[1:] 
    raw_data = df_data[cols].T.values


    # Step 1: Get train, val, and test splits
    # For now, we use SplitterByTimestamp only
    train_data, val_data, test_data, buff = SplitterByTimestamp(
        seq_len,
        pred_len,
        train_ratio=test_train_ratio,
        val_ratio=test_val_ratio,
        prompt_folder_path=prompt_data_path,
        data_name=test_data_path
    ).get_csv_splits(df_data)



    # Step 2: Scale the datasets. We fit on the whole sequence by default.
    # To fit on the train sequence only, set scale_on_train=True
    # For now, we use StandardScaler only
    processor = processor_dict[data_processing]() 
    train_data, val_data, test_data = processor.process(
        raw_data,  # Used for scaling
        train_data,
        val_data,
        test_data,
        fit_train_only=scale_on_train,
    )

    test_prompt_data = []
    test_prompt_data_path = prompt_data_path + "/test"
    for test_instance_idx in buff:
            instance_prompt = _get_csv_prompt(
                test_prompt_data_path,  
                test_data_path,
                test_instance_idx
            )
            test_prompt_data.append(instance_prompt)
    test_dataset = TSPromptDataset (
        data=test_data, 
        prompt=test_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=1,
        uniform_sampling=False
    )
    return test_dataset, processor

def create_csv_statprompt_datasets(
    data_path,
    prompt_data_path,
    data_processing,
    seq_len,
    pred_len,
    train_ratio,
    val_ratio,
    scale_on_train=False,
    downsample_rate=10,
):
    # Load Training data and validation data
    train_data, val_data, test_data, train_prompt_data, val_prompt_data, test_prompt_data = [], [], [], [], [], []
    for sub_data_path in data_path:
        sub_train_ratio = train_ratio
        sub_val_ratio = val_ratio
        
        # Step 0: Read data, the output is a list of 1-d time-series
        df_data = pd.read_csv(sub_data_path)
        cols = df_data.columns[1:] 
        raw_data = df_data[cols].T.values
        
        # keep the ETT data setting same with PatchTST (dataset length and train/val/test ratio)
        if 'ETTh1' in sub_data_path or 'ETTh2' in sub_data_path:
            sub_train_ratio = 0.6
            sub_val_ratio = 0.2
        if 'ETTm1' in sub_data_path or 'ETTm2' in sub_data_path:
            sub_train_ratio = 0.6
            sub_val_ratio = 0.2
        test_ratio = 1.0 - sub_train_ratio - sub_val_ratio
        print(f"Training Loading {sub_data_path}, train {sub_train_ratio:.2f}, val {sub_val_ratio:.2f}, test {test_ratio:.2f}")


        # Step 1: Get train, val, and test splits
        # For now, we use SplitterByTimestamp only
        sub_train_data, sub_val_data, sub_test_data, buff = SplitterByTimestamp(
            seq_len,
            pred_len,
            train_ratio=sub_train_ratio,
            val_ratio=sub_val_ratio,
            prompt_folder_path=prompt_data_path,
            data_name=sub_data_path
        ).get_csv_splits(df_data)


        # Step 2: Scale the datasets. We fit on the whole sequence by default.
        # To fit on the train sequence only, set scale_on_train=True
        # For now, we use StandardScaler only
        processor = processor_dict[data_processing]() 
        sub_train_data, sub_val_data, sub_test_data = processor.process(
            raw_data,  # Used for scaling
            sub_train_data,
            sub_val_data,
            sub_test_data,
            fit_train_only=scale_on_train,
        )

        # Step 2.5 Load prompt for each instance
        # Train Prompt
        train_prompt_data_path = prompt_data_path + '/train'
        for train_instance_idx in buff:
            instance_prompt =_get_csv_prompt(
                train_prompt_data_path,  
                sub_data_path,
                train_instance_idx
            )
            train_prompt_data.append(instance_prompt)
        
        val_prompt_data_path = prompt_data_path + '/val'
        for val_instance_idx in buff:
            instance_prompt = _get_csv_prompt(
                val_prompt_data_path,  
                sub_data_path,
                val_instance_idx
            )
            val_prompt_data.append(instance_prompt)
        
        train_data.extend(sub_train_data)
        val_data.extend(sub_val_data)


    # Step 3: Create Torch datasets (samplers)
    train_dataset = TSPromptDataset (
        data=train_data,
        prompt=train_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
    )

    val_dataset = TSPromptDataset (
        data=val_data,
        prompt=val_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
    )
    

    return train_dataset, val_dataset, processor

def create_csv_statprompt_test_datasets(
    test_data_path,
    prompt_data_path,
    data_processing,
    seq_len,
    pred_len,
    train_ratio,
    val_ratio,
    scale_on_train=False,
    downsample_rate=10,
):
        # Load Testing data
    test_train_ratio, test_val_ratio = train_ratio, val_ratio

    if 'ETTh1' in test_data_path or 'ETTh2' in test_data_path:
        test_train_ratio = 0.6
        test_val_ratio = 0.2
    if 'ETTm1' in test_data_path or 'ETTm2' in test_data_path:
        test_train_ratio = 0.6
        test_val_ratio = 0.2
    test_test_ratio = 1.0 - test_train_ratio - test_val_ratio
    print(f"Testing Loading {test_data_path}, train {test_train_ratio:.2f}, val {test_val_ratio:.2f}, test {test_test_ratio:.2f}")

    # Step 0: Read data, the output is a list of 1-d time-series
    df_data = pd.read_csv(test_data_path)
    cols = df_data.columns[1:] 
    raw_data = df_data[cols].T.values


    # Step 1: Get train, val, and test splits
    # For now, we use SplitterByTimestamp only
    train_data, val_data, test_data, buff = SplitterByTimestamp(
        seq_len,
        pred_len,
        train_ratio=test_train_ratio,
        val_ratio=test_val_ratio,
        prompt_folder_path=prompt_data_path,
        data_name=test_data_path
    ).get_csv_splits(df_data)



    # Step 2: Scale the datasets. We fit on the whole sequence by default.
    # To fit on the train sequence only, set scale_on_train=True
    # For now, we use StandardScaler only
    processor = processor_dict[data_processing]() 
    train_data, val_data, test_data = processor.process(
        raw_data,  # Used for scaling
        train_data,
        val_data,
        test_data,
        fit_train_only=scale_on_train,
    )

    test_prompt_data = []
    test_prompt_data_path = prompt_data_path + "/test"
    for test_instance_idx in buff:
            instance_prompt = _get_csv_prompt(
                test_prompt_data_path,  
                test_data_path,
                test_instance_idx
            )
            test_prompt_data.append(instance_prompt)
            
    test_dataset = TSPromptDataset (
        data=test_data,
        prompt=test_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=1,
    )
    return test_dataset, processor

def create_csv_textprompt_datasets(
    data_path,
    prompt_data_path,
    data_processing,
    seq_len,
    pred_len,
    train_ratio,
    val_ratio,
    scale_on_train=False,
    downsample_rate=10,
):

    # Load Training data and validation data
    train_data, val_data, test_data, train_prompt_data, val_prompt_data, test_prompt_data = [], [], [], [], [], []

    for sub_data_path in data_path:  
        # Step 0: Read data, the output is a list of 1-d time-series      
        data_name = sub_data_path.split('/')[-1].split('.')[0]
        sub_train_ratio = train_ratio
        sub_val_ratio = val_ratio
        
        df_data = pd.read_csv(sub_data_path)
        cols = df_data.columns[1:]
        raw_data = df_data[cols].T.values
        
        # keep the ETT data setting same with PatchTST (dataset length and train/val/test ratio)
        if 'ETTh1' in sub_data_path or 'ETTh2' in sub_data_path:
            sub_train_ratio = 0.6
            sub_val_ratio = 0.2
        if 'ETTm1' in sub_data_path or 'ETTm2' in sub_data_path:
            sub_train_ratio = 0.6
            sub_val_ratio = 0.2
        test_ratio = 1.0 - sub_train_ratio - sub_val_ratio
        print(f"Training Loading {sub_data_path}, train {sub_train_ratio:.2f}, val {sub_val_ratio:.2f}, test {test_ratio:.2f}")

        # Step 1: Get train, val, and test splits
        # For now, we use SplitterByTimestamp only
        sub_train_data, sub_val_data, sub_test_data, buff = SplitterByTimestamp(
            seq_len,
            pred_len,
            train_ratio=sub_train_ratio,
            val_ratio=sub_val_ratio,
            prompt_folder_path=prompt_data_path,
            data_name=sub_data_path
        ).get_csv_splits(df_data)


        # Step 2: Scale the datasets. We fit on the whole sequence by default.
        # To fit on the train sequence only, set scale_on_train=True
        # For now, we use StandardScaler only
        processor = processor_dict[data_processing]() 
        sub_train_data, sub_val_data, sub_test_data = processor.process(
            raw_data,  # Used for scaling
            sub_train_data,
            sub_val_data,
            sub_test_data,
            fit_train_only=scale_on_train,
        )

        # Step 2.5 Load index of every data class for each instance, as prompt data will be different for different datasets
        # Train Prompt
        for _ in buff:
            train_prompt_data.append([data2index[data_name]])
        
        for _ in buff:
            val_prompt_data.append([data2index[data_name]])

        # Merge the list of data        
        train_data.extend(sub_train_data)
        val_data.extend(sub_val_data)


    # Step 3: Create Torch datasets (samplers)
    train_dataset = TSPromptDataset(
        data=train_data,
        prompt=train_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
    )

    val_dataset = TSPromptDataset(
        data=val_data,
        prompt=val_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
    )
 

    return train_dataset, val_dataset, processor

def create_csv_textprompt_test_datasets(
    test_data_path,
    prompt_data_path,
    data_processing,
    seq_len,
    pred_len,
    train_ratio,
    val_ratio,
    scale_on_train=False,
    downsample_rate=10,
):
    # Load Testing data
    test_prompt_data = []
    test_train_ratio, test_val_ratio = train_ratio, val_ratio

    if 'ETTh1' in test_data_path or 'ETTh2' in test_data_path:
        test_train_ratio = 0.6
        test_val_ratio = 0.2
    if 'ETTm1' in test_data_path or 'ETTm2' in test_data_path:
        test_train_ratio = 0.6
        test_val_ratio = 0.2
    test_test_ratio = 1.0 - test_train_ratio - test_val_ratio
    print(f"Testing Loading {test_data_path}, train {test_train_ratio:.2f}, val {test_val_ratio:.2f}, test {test_test_ratio:.2f}")


    # Step 0: Read data, the output is a list of 1-d time-series
    test_data_name = test_data_path.split('/')[-1].split('.')[0]
    df_data = pd.read_csv(test_data_path)
    cols = df_data.columns[1:] 
    raw_data = df_data[cols].T.values

    # Step 1: Get train, val, and test splits
    # For now, we use SplitterByTimestamp only
    train_data, val_data, test_data, buff = SplitterByTimestamp(
        seq_len,
        pred_len,
        train_ratio=test_train_ratio,
        val_ratio=test_val_ratio,
        prompt_folder_path=prompt_data_path,
        data_name=test_data_path
    ).get_csv_splits(df_data)


    # Step 2: Scale the datasets. We fit on the whole sequence by default.
    # To fit on the train sequence only, set scale_on_train=True
    # For now, we use StandardScaler only
    processor = processor_dict[data_processing]() 
    train_data, val_data, test_data = processor.process(
        raw_data,  # Used for scaling
        train_data,
        val_data,
        test_data,
        fit_train_only=scale_on_train,
    )
    
    # Step 2.5 Load index of every data class for each instance, as prompt data will be different for different datasets
    for _ in buff:
        test_prompt_data.append([data2index[test_data_name]])
        
    test_dataset = TSPromptDataset(
        data=test_data, # add 1 dimension to match the dimension of training data in dataloader
        prompt=test_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=1,
    )
    return test_dataset, processor

def create_csv_token_datasets(
    data_path,
    prompt_data_path,
    data_processing,
    seq_len,
    pred_len,
    train_ratio,
    val_ratio,
    scale_on_train=False,
    downsample_rate=10,
):
    # Load Training data and validation data
    train_data, val_data, test_data, train_prompt_data, val_prompt_data, test_prompt_data = [], [], [], [], [], []
    for sub_data_path in data_path:
        sub_train_ratio = train_ratio
        sub_val_ratio = val_ratio
        
        # Step 0: Read data, the output is a list of 1-d time-series
        df_data = pd.read_csv(sub_data_path)
        cols = df_data.columns[1:] 
        raw_data = df_data[cols].T.values
        
        # keep the ETT data setting same with PatchTST (dataset length and train/val/test ratio)
        if 'ETTh1' in sub_data_path or 'ETTh2' in sub_data_path:
            sub_train_ratio = 0.6
            sub_val_ratio = 0.2
        if 'ETTm1' in sub_data_path or 'ETTm2' in sub_data_path:
            sub_train_ratio = 0.6
            sub_val_ratio = 0.2
        test_ratio = 1.0 - sub_train_ratio - sub_val_ratio
        print(f"Training Loading {sub_data_path}, train {sub_train_ratio:.2f}, val {sub_val_ratio:.2f}, test {test_ratio:.2f}")


        # Step 1: Get train, val, and test splits
        # For now, we use SplitterByTimestamp only
        sub_train_data, sub_val_data, sub_test_data, buff = SplitterByTimestamp(
            seq_len,
            pred_len,
            train_ratio=sub_train_ratio,
            val_ratio=sub_val_ratio,
            prompt_folder_path=prompt_data_path,
            data_name=sub_data_path
        ).get_csv_splits(df_data)


        # Step 2: Scale the datasets. We fit on the whole sequence by default.
        # To fit on the train sequence only, set scale_on_train=True
        # For now, we use StandardScaler only
        processor = processor_dict[data_processing]() 
        sub_train_data, sub_val_data, sub_test_data = processor.process(
            raw_data,  # Used for scaling
            sub_train_data,
            sub_val_data,
            sub_test_data,
            fit_train_only=scale_on_train,
        )

        # Step 2.5 Load prompt for each instance
        # Train Prompt
        train_prompt_data_path = prompt_data_path + '/train'
        for train_instance_idx in buff:
            instance_prompt =_get_csv_prompt(
                train_prompt_data_path,  
                sub_data_path,
                train_instance_idx
            )
            train_prompt_data.append(instance_prompt)
        
        val_prompt_data_path = prompt_data_path + '/val'
        for val_instance_idx in buff:
            instance_prompt = _get_csv_prompt(
                val_prompt_data_path,  
                sub_data_path,
                val_instance_idx
            )
            val_prompt_data.append(instance_prompt)
        
        train_data.extend(sub_train_data)
        val_data.extend(sub_val_data)


    # Step 3: Create Torch datasets (samplers)
    train_dataset = TSTokenDataset (
        data=train_data,
        prompt=train_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
    )

    val_dataset = TSTokenDataset (
        data=val_data,
        prompt=val_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
    )
    

    return train_dataset, val_dataset, processor

def create_csv_token_test_datasets(
    test_data_path,
    prompt_data_path,
    data_processing,
    seq_len,
    pred_len,
    train_ratio,
    val_ratio,
    scale_on_train=False,
    downsample_rate=10,
):
        # Load Testing data
    test_train_ratio, test_val_ratio = train_ratio, val_ratio

    if 'ETTh1' in test_data_path or 'ETTh2' in test_data_path:
        test_train_ratio = 0.6
        test_val_ratio = 0.2
    if 'ETTm1' in test_data_path or 'ETTm2' in test_data_path:
        test_train_ratio = 0.6
        test_val_ratio = 0.2
    test_test_ratio = 1.0 - test_train_ratio - test_val_ratio
    print(f"Testing Loading {test_data_path}, train {test_train_ratio:.2f}, val {test_val_ratio:.2f}, test {test_test_ratio:.2f}")

    # Step 0: Read data, the output is a list of 1-d time-series
    df_data = pd.read_csv(test_data_path)
    cols = df_data.columns[1:] 
    raw_data = df_data[cols].T.values


    # Step 1: Get train, val, and test splits
    # For now, we use SplitterByTimestamp only
    train_data, val_data, test_data, buff = SplitterByTimestamp(
        seq_len,
        pred_len,
        train_ratio=test_train_ratio,
        val_ratio=test_val_ratio,
        prompt_folder_path=prompt_data_path,
        data_name=test_data_path
    ).get_csv_splits(df_data)



    # Step 2: Scale the datasets. We fit on the whole sequence by default.
    # To fit on the train sequence only, set scale_on_train=True
    # For now, we use StandardScaler only
    processor = processor_dict[data_processing]() 
    train_data, val_data, test_data = processor.process(
        raw_data,  # Used for scaling
        train_data,
        val_data,
        test_data,
        fit_train_only=scale_on_train,
    )

    test_prompt_data = []
    test_prompt_data_path = prompt_data_path + "/test"
    for test_instance_idx in buff:
            instance_prompt = _get_csv_prompt(
                test_prompt_data_path,  
                test_data_path,
                test_instance_idx
            )
            test_prompt_data.append(instance_prompt)
            
    test_dataset = TSTokenDataset (
        data=test_data,
        prompt=test_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=1,
    )
    return test_dataset, processor

def create_datasets(
    data_path,
    prompt_data_path,
    data_processing,
    seq_len,
    pred_len,
    prompt_len,
    train_ratio,
    val_ratio,
    scale_on_train=False,
    downsample_rate=10,
):
    # Load Training data and validation data
    train_data, val_data, test_data, prompt_data = [], [], [], []

    for sub_data_path in data_path:
        test_ratio = 1.0 - train_ratio - val_ratio
        print(f"Training Loading {sub_data_path}, train {train_ratio:.2f}, val {val_ratio:.2f}, test {test_ratio:.2f}")

        # We parse the datapath to get the dataset class
        dir_name = os.path.split(os.path.dirname(sub_data_path))[-1]
        
        # Step 0: Read data, the output is a list of 1-d time-series
        raw_data = reader_dict[dir_name](sub_data_path).fetch()

        # Step 1: Get train, val, and test splits
        # For now, we use SplitterByTimestamp only
        sub_train_data, sub_val_data, sub_test_data, buff = SplitterByTimestamp(
            seq_len,
            pred_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            prompt_folder_path=prompt_data_path,
            data_name=sub_data_path
        ).get_splits(raw_data)



        # Step 2: Scale the datasets. We fit on the whole sequence by default.
        # To fit on the train sequence only, set scale_on_train=True
        # For now, we use StandardScaler only
        processor = processor_dict[data_processing]() 
        sub_train_data, sub_val_data, sub_test_data = processor.process(
            raw_data,  # Used for scaling
            sub_train_data,
            sub_val_data,
            sub_test_data,
            fit_train_only=scale_on_train,
        )

        # Step 2.5 Load prompt for each instance
        for instance_idx in buff:
            instance_prompt = _get_prompt(
                prompt_data_path,  
                sub_data_path,
                instance_idx
            )
            prompt_data.append(instance_prompt)

        # Step 2.5: Merge the list of data
        train_data.extend(sub_train_data)
        val_data.extend(sub_val_data)
        test_data.extend(sub_test_data)
        
    # Step 3: Create Torch datasets (samplers)
    train_dataset = TSPromptDataset(
        data=train_data,
        prompt=prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
    )

    val_dataset = TSPromptDataset(
        data=val_data,
        prompt=prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=54,
    )

    return train_dataset, val_dataset, processor

def create_test_datasets(
    test_data_path,
    prompt_data_path,
    data_processing,
    seq_len,
    pred_len,
    prompt_len,
    train_ratio,
    val_ratio,
    scale_on_train=False,
    downsample_rate=10,
):
        # Testing data
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"Testing Loading {test_data_path}, train {train_ratio:.2f}, val {val_ratio:.2f}, test {test_ratio:.2f}")

    # We parse the datapath to get the dataset class
    dir_name = os.path.split(os.path.dirname(test_data_path))[-1]

    # Step 0: Read data, the output is a list of 1-d time-series
    raw_data = reader_dict[dir_name](test_data_path).fetch()

    # Step 1: Get train, val, and test splits
    # For now, we use SplitterByTimestamp only
    train_data, val_data, test_data, buff = SplitterByTimestamp(
        seq_len,
        pred_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        prompt_folder_path=prompt_data_path,
        data_name=test_data_path
    ).get_splits(raw_data)

    # Step 2: Scale the datasets. We fit on the whole sequence by default.
    # To fit on the train sequence only, set scale_on_train=True
    # For now, we use StandardScaler only
    processor = processor_dict[data_processing]() 
    train_data, val_data, test_data = processor.process(
        raw_data,  # Used for scaling
        train_data,
        val_data,
        test_data,
        fit_train_only=scale_on_train,
    )

    prompt_data = []
    for instance_idx in buff:
            instance_prompt = _get_prompt(
                prompt_data_path,  
                test_data_path,
                instance_idx
            )
            prompt_data.append(instance_prompt)

    test_dataset = TSPromptDataset(
        data=[test_data], # add 1 dimension to match the dimension of training data in dataloader
        prompt=prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=1,
        uniform_sampling=False
    )
    return test_dataset, processor

def _get_prompt(prompt_folder_path, data_name, idx_file_name):
    prompt_name = data_name.split("/")[-1]
    prompt_name = prompt_name.replace(".tsf", "")
    prompt_path = os.path.join(prompt_folder_path, prompt_name, "T"+str(idx_file_name+1)+"_prompt.pth.tar")
    if not os.path.exists(prompt_path):
        return
    prompt_data = torch.load(prompt_path)
    prompt_data = prompt_data.T[0]
    
    # ipdb.set_trace()
    prompt_data = [ prompt_data.iloc[i] for i in range(len(prompt_data)) ]
    return prompt_data

def _get_csv_prompt(prompt_folder_path, data_name, idx_file_name):
    data_path = data_name.split('/')[-2]+'/'+data_name.split('/')[-1].split('.')[0]
    idx_file_name = idx_file_name.replace("/", "-")
    idx_file_name = idx_file_name.replace("**", "_")
    idx_file_name = idx_file_name.replace("%", "_")
    
    prompt_path = os.path.join(prompt_folder_path,data_path+'_'+str(idx_file_name)+"_prompt.pth.tar")
    if not os.path.exists(prompt_path):
        print(f"Prompt file {prompt_path} does not exist")
        return
    
    prompt_data = torch.load(prompt_path)
    prompt_data = prompt_data.T[0]

    prompt_data = [ prompt_data.iloc[i] for i in range(len(prompt_data)) ]
    return prompt_data

def get_datasets(args): 
    ds_factory = DatasetFactory(
        data_paths=args.data_path,
        prompt_data_path=args.prompt_data_path,
        data_processing=args.data_processing,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        model=args.model,
        downsample_rate=args.downsample_rate
    )
    train_ds, val_ds, test_ds_list, processor = ds_factory.getDatasets()

    # print(f"Data loaded {args.test_data_path}, train size {len(train_dataset)}, val size {len(val_dataset)}, test size {len(test_dataset)} ")
    return train_ds, val_ds, processor
    
# def get_test_datasets(args):
#     # Get datasets extension
#     file_ext = os.path.splitext(args.data_path[0])[-1]
    
#     # use CSV datasets
#     if file_ext == ".csv":
#         if "WordPrompt" in args.model:
#             test_dataset, processor = create_csv_textprompt_test_datasets(
#                 test_data_path=args.test_data_path,
#                 prompt_data_path=args.prompt_data_path,
#                 data_processing=args.data_processing,
#                 seq_len=args.seq_len,
#                 pred_len=args.pred_len,
#                 train_ratio=args.train_ratio,
#                 val_ratio=args.val_ratio,
#                 downsample_rate=args.downsample_rate,
#             )
#         elif "Tokenizer" in args.model:
#             test_dataset, processor = create_csv_token_test_datasets(
#                 test_data_path=args.test_data_path,
#                 prompt_data_path=args.prompt_data_path,
#                 data_processing=args.data_processing,
#                 seq_len=args.seq_len,
#                 pred_len=args.pred_len,
#                 train_ratio=args.train_ratio,
#                 val_ratio=args.val_ratio,
#             )
#         else:
#             test_dataset, processor = create_csv_statprompt_test_datasets(
#                 test_data_path=args.test_data_path,
#                 prompt_data_path=args.prompt_data_path,
#                 data_processing=args.data_processing,
#                 seq_len=args.seq_len,
#                 pred_len=args.pred_len,
#                 train_ratio=args.train_ratio,
#                 val_ratio=args.val_ratio,
#                 downsample_rate=args.downsample_rate,
#             )
#     else:
#         # use Monash datasets
#         test_dataset, processor = create_datasets(
#             test_data_path=args.test_data_path,
#             prompt_data_path=args.prompt_data_path,
#             data_processing=args.data_processing,
#             seq_len=args.seq_len,
#             pred_len=args.pred_len,
#             train_ratio=args.train_ratio,
#             val_ratio=args.val_ratio,
#             downsample_rate=args.downsample_rate,
#         )
#     print(f"Data loaded {args.test_data_path}, test size {len(test_dataset)} ")
#     return test_dataset, processor    

def get_data_loaders(args):

    # Create datasets
    train_dataset, val_dataset, test_dataset, processor = create_datasets(
        data_path=args.data_path,
        data_processing=args.data_processing,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"Data loaded, train size {len(train_dataset)}, val size {len(val_dataset)}, test size {len(test_dataset)}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader, processor