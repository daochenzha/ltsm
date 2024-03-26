import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tsbench.data_pipeline.reader import reader_dict
from tsbench.data_pipeline.splitter import SplitterByTimestamp
from tsbench.data_pipeline.data_processing import processor_dict
from tsbench.data_pipeline.dataset import TSDataset,  TSPromptDataset, TSPromptDatasetTXT
import ipdb

def create_csv_datasets(
    data_path,
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
    # Here, we directly load the training, validation, and testing splits
    # to aviod loading the same dataset 3 times

    # TODO: we can support using only a percentage of the training data
    # However, this could be confusing. Thus, we will only do it when needed
    if 'text' in prompt_data_path:
        get_prompt = _get_txt_prompt
        TimeSeriesDataset = TSPromptDatasetTXT
    else:
        get_prompt = _get_csv_prompt
        TimeSeriesDataset = TSPromptDataset

        
    # Training data
    train_data, val_data, test_data, train_prompt_data, val_prompt_data, test_prompt_data = [], [], [], [], [], []
    train_data1, val_data1, test_data1, prompt_data1 = [],[],[],[]
    for sub_data_path in data_path:
        

        # We parse the datapath to get the dataset class
        # dir_name = os.path.split(os.path.dirname(sub_data_path))[-1]
        sub_train_ratio = train_ratio
        sub_val_ratio = val_ratio
        
        df_data = pd.read_csv(sub_data_path)
        cols = df_data.columns[1:] 
        raw_data = df_data[cols].T.values
        
        # keep the ETT data setting same with PatchTST (dataset length and train/val/test ratio)
        if 'ETTh1' in sub_data_path or 'ETTh2' in sub_data_path:
            # raw_data = df_data[cols][:14400].T.values
            sub_train_ratio = 0.6
            sub_val_ratio = 0.2
        if 'ETTm1' in sub_data_path or 'ETTm2' in sub_data_path:
            # raw_data = df_data[cols][:57600].T.values
            sub_train_ratio = 0.6
            sub_val_ratio = 0.2
        test_ratio = 1.0 - sub_train_ratio - sub_val_ratio
        print(f"Training Loading {sub_data_path}, train {sub_train_ratio:.2f}, val {sub_val_ratio:.2f}, test {test_ratio:.2f}")

        # Step 0: Read data, the output is a list of 1-d time-series
        # raw_data = reader_dict[dir_name](sub_data_path).fetch()

        # ipdb.set_trace()
        """
        print(len(raw_data))
        print(raw_data[0])
        print(raw_data[0].shape)
        for a in raw_data:
            print(len(a))
        exit()
        """

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

        """
        print(len(train_data), train_data[0].shape)
        print(len(val_data), val_data[0].shape)
        print(len(test_data), test_data[0].shape)
        exit()
        """

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
        for train_intance_idx in buff:
            instance_prompt =get_prompt(
                train_prompt_data_path,  
                sub_data_path,
                train_intance_idx
            )
            train_prompt_data.append(instance_prompt)
        
        val_prompt_data_path = prompt_data_path + '/val'
        for val_intance_idx in buff:
            instance_prompt = get_prompt(
                val_prompt_data_path,  
                sub_data_path,
                val_intance_idx
            )
            val_prompt_data.append(instance_prompt)

        # Step 2.5: Merge the list of data
        train_data.extend(sub_train_data)
        val_data.extend(sub_val_data)
        test_data.extend(sub_test_data)
        
        train_data1.append(sub_train_data)
        val_data1.append(sub_val_data)
        test_data1.append(sub_test_data)


        # ipdb.set_trace()

    # Step 3: Create Torch datasets (samplers)
    train_dataset = TimeSeriesDataset(
        data=train_data1,
        prompt=train_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
        uniform_sampling=False
    )

    val_dataset = TimeSeriesDataset(
        data=val_data1,
        prompt=val_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
        uniform_sampling=False
    )
    
    test_train_ratio, test_val_ratio = train_ratio, val_ratio
    # Testing data
    if 'ETTh1' in test_data_path or 'ETTh2' in test_data_path:
        # raw_data = df_data[cols][:14400].T.values
        test_train_ratio = 0.6
        test_val_ratio = 0.2
    if 'ETTm1' in test_data_path or 'ETTm2' in test_data_path:
        # raw_data = df_data[cols][:57600].T.values
        test_train_ratio = 0.6
        test_val_ratio = 0.2
    test_test_ratio = 1.0 - test_train_ratio - test_val_ratio
    print(f"Testing Loading {test_data_path}, train {test_train_ratio:.2f}, val {test_val_ratio:.2f}, test {test_test_ratio:.2f}")

    # We parse the datapath to get the dataset class
    # dir_name = os.path.split(os.path.dirname(test_data_path))[-1]

    # Step 0: Read data, the output is a list of 1-d time-series
    df_data = pd.read_csv(test_data_path)
    cols = df_data.columns[1:] 
    raw_data = df_data[cols].T.values
    """
    print(len(raw_data))
    print(raw_data[0])
    print(raw_data[0].shape)
    for a in raw_data:
        print(len(a))
    exit()
    """

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

    """
    print(len(train_data), train_data[0].shape)
    print(len(val_data), val_data[0].shape)
    print(len(test_data), test_data[0].shape)
    exit()
    """

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

    """
    print(len(train_data), train_data[0].shape, train_data[3])
    print(len(val_data), val_data[0].shape, val_data[3])
    print(len(test_data), test_data[0].shape, test_data[3])
    exit()
    """
    test_prompt_data = []
    test_prompt_data_path = prompt_data_path + "/test"
    for test_intance_idx in buff:
            instance_prompt = get_prompt(
                test_prompt_data_path,  
                test_data_path,
                test_intance_idx
            )
            test_prompt_data.append(instance_prompt)
    # TODO : Fix this
    test_dataset = TimeSeriesDataset(
        data=[test_data], # add 1 dimension to match the dimension of training data in dataloader
        prompt=test_prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=1,
        uniform_sampling=False
    )

    return train_dataset, val_dataset,test_dataset, processor

def create_datasets(
    data_path,
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
    # Here, we directly load the training, validation, and testing splits
    # to aviod loading the same dataset 3 times

    # TODO: we can support using only a percentage of the training data
    # However, this could be confusing. Thus, we will only do it when needed

    # Training data
    train_data, val_data, test_data, prompt_data = [], [], [], []
    train_data1, val_data1, test_data1, prompt_data1 = [],[],[],[]
    for sub_data_path in data_path:
        test_ratio = 1.0 - train_ratio - val_ratio
        print(f"Training Loading {sub_data_path}, train {train_ratio:.2f}, val {val_ratio:.2f}, test {test_ratio:.2f}")

        # We parse the datapath to get the dataset class
        dir_name = os.path.split(os.path.dirname(sub_data_path))[-1]
        
        # Step 0: Read data, the output is a list of 1-d time-series
        raw_data = reader_dict[dir_name](sub_data_path).fetch()
        # ipdb.set_trace()
        """
        print(len(raw_data))
        print(raw_data[0])
        print(raw_data[0].shape)
        for a in raw_data:
            print(len(a))
        exit()
        """

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

        """
        print(len(train_data), train_data[0].shape)
        print(len(val_data), val_data[0].shape)
        print(len(test_data), test_data[0].shape)
        exit()
        """

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
        for intance_idx in buff:
            instance_prompt = _get_prompt(
                prompt_data_path,  
                sub_data_path,
                intance_idx
            )
            prompt_data.append(instance_prompt)

        # Step 2.5: Merge the list of data
        train_data.extend(sub_train_data)
        val_data.extend(sub_val_data)
        test_data.extend(sub_test_data)
        
        train_data1.append(sub_train_data)
        val_data1.append(sub_val_data)
        test_data1.append(sub_test_data)


        # ipdb.set_trace()

    # Step 3: Create Torch datasets (samplers)
    train_dataset = TSPromptDataset(
        data=train_data1,
        prompt=prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=downsample_rate,
    )

    val_dataset = TSPromptDataset(
        data=val_data1,
        prompt=prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=54,
    )
    
    # Testing data
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"Testing Loading {test_data_path}, train {train_ratio:.2f}, val {val_ratio:.2f}, test {test_ratio:.2f}")

    # We parse the datapath to get the dataset class
    dir_name = os.path.split(os.path.dirname(test_data_path))[-1]

    # Step 0: Read data, the output is a list of 1-d time-series
    raw_data = reader_dict[dir_name](test_data_path).fetch()
    """
    print(len(raw_data))
    print(raw_data[0])
    print(raw_data[0].shape)
    for a in raw_data:
        print(len(a))
    exit()
    """

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

    """
    print(len(train_data), train_data[0].shape)
    print(len(val_data), val_data[0].shape)
    print(len(test_data), test_data[0].shape)
    exit()
    """

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

    """
    print(len(train_data), train_data[0].shape, train_data[3])
    print(len(val_data), val_data[0].shape, val_data[3])
    print(len(test_data), test_data[0].shape, test_data[3])
    exit()
    """
    prompt_data = []
    for intance_idx in buff:
            instance_prompt = _get_prompt(
                prompt_data_path,  
                test_data_path,
                intance_idx
            )
            prompt_data.append(instance_prompt)
    # TODO : Fix this
    test_dataset = TSPromptDataset(
        data=[test_data], # add 1 dimension to match the dimension of training data in dataloader
        prompt=prompt_data,
        seq_len=seq_len,
        pred_len=pred_len,
        downsample_rate=1,
        uniform_sampling=False
    )

    return train_dataset, val_dataset, test_dataset, processor

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
    prompt_name = data_name.split("/")[-1]
    prompt_name = prompt_name.replace(".tsf", "")
    data_path = data_name.split('/')[-2]+'/'+data_name.split('/')[-1].split('.')[0]
    idx_file_name = idx_file_name.replace("/", "-")
    
    prompt_path = os.path.join(prompt_folder_path,data_path+'_'+str(idx_file_name)+"_prompt.pth.tar")
    if not os.path.exists(prompt_path):
        print(f"Prompt file {prompt_path} does not exist")
        return
    
    prompt_data = torch.load(prompt_path)
    
    prompt_data = prompt_data.T[0]
    
    # ipdb.set_trace()
    prompt_data = [ prompt_data.iloc[i] for i in range(len(prompt_data)) ]
    return prompt_data

def _get_txt_prompt(prompt_folder_path, data_name, idx_file_name):
    prompt_name = data_name.split("/")[-1]
    prompt_name = prompt_name.replace(".tsf", "")
    data_path = data_name.split('/')[-1].split('.')[0]
    prompt_folder_path,data_path = "/home/sl237/ltsm/ltsm/dataset/prompt_bank/"
    txt_prompt_path = os.path.join(prompt_folder_path,data_path, data_path+".txt")
    if not os.path.exists(txt_prompt_path):
        print(f"Prompt file {txt_prompt_path} does not exist")
        return
    
    with open(txt_prompt_path, 'r') as file:
        txt_prompt = file.read()

    return txt_prompt


def get_datasets(args):
    file_ext = os.path.splitext(args.data_path[0])[-1]
    if file_ext == ".csv":
        train_dataset, val_dataset, test_dataset, processor = create_csv_datasets(
            data_path=args.data_path,
            test_data_path=args.test_data_path,
            prompt_data_path=args.prompt_data_path,
            data_processing=args.data_processing,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            prompt_len=args.prompt_len,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            downsample_rate=args.downsample_rate,
        )
    else:
    # Create datasets
        train_dataset, val_dataset, test_dataset, processor = create_datasets(
            data_path=args.data_path,
            test_data_path=args.test_data_path,
            prompt_data_path=args.prompt_data_path,
            data_processing=args.data_processing,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            prompt_len=args.prompt_len,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            downsample_rate=args.downsample_rate,
        )
    # print(f"Data loaded {args.test_data_path}, train size {len(train_dataset)}, val size {len(val_dataset)}, test size {len(test_dataset)}")
    print(f"Data loaded {args.test_data_path}, train size {len(train_dataset)}, val size {len(val_dataset)}, test size {len(test_dataset)} ")
    return train_dataset, val_dataset,test_dataset, processor
    
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

