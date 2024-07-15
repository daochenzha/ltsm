# Tutorial of LTSM-bundle


## Installation
```
conda create -n ltsm python=3.8.0
conda activate ltsm
git clone git@github.com:daochenzha/ltsm.git
cd ltsm
pip3 install -e .
pip3 install -r requirements.txt
```


## :bookmark: Step 0: Collect Datasets and Time Series Prompts

### :cyclone: You can use our prepared dataset to on-board youselves on LTSM-bundle

### Download training datasets
```bash
cd datasets
download: https://drive.google.com/drive/folders/1hLFbz0FRxdiDCzgFYtKCOPJYSBVvwW9P
```

### Download time sereis prompts 
```bash
cd prompt_bank/propmt_data_csv
download: https://drive.google.com/drive/folders/1hLFbz0FRxdiDCzgFYtKCOPJYSBVvwW9P
```

### Check word prompts 
```bash
cd prompt_bank/text_prompt_data_csv/
check: csv_prompt.json
```

## :bookmark: Step 1: Customize Datasets and Time Series Prompts

### :cyclone: If you prefer to build LTSM-bundle on your own dataset, please follow the 5-step instructions below:

**Step 1-a.** Prepare your dataset. Make sure your local data folder like this:
````angular2html
- ltsm/
    - datasets/
        DATA_1.csv/
        DATA_2.csv/
    ...
````

**Step 1-b.** Generating the time series prompts from training, validating, and testing datasets
````angular2html
python3 prompt_generate_split.py
````

**Step 1-c.** Find the generated time series prompts in the './prompt_data_split' folder. Then run the following command for normalizing the prompts:
````angular2html
python3 prompt_normalization_split.py --mode fit
````

**Step 1-d.** Run this command to export the prompts to the "./prompt_data_normalize_split" folder:
````angular2html
python3 prompt_normalization_split.py --mode transform
````

**Step 1-e.** Modify the word prompt based on your dataset description in "prompt_bank/text_prompt_data_csv/csv_prompt.json":
````angular2html
vim prompt_bank/text_prompt_data_csv/csv_prompt.json
````

## :bookmark: Step 2: Customize your own LTSM-bundle 

### :cyclone: Now, it's time to build you own LTSM-bundle!!

#### (1) Explore [Word Prompt] and [Linear Tokenization] on gpt2-medium
```bash
python3 main_ltsm.py \
    --model LTSM_WordPrompt \
    --model_name_or_path gpt2-medium \
    --train_epochs 500 \
    --batch_size 10 \
    --pred_len 96 \
    --data_path "datasets/ETT-small/ETTh1.csv" \
    --test_data_path_list "datasets/ETT-small/ETTh1.csv" \
    --prompt_data_path "prompt_bank/text_prompt_data_csv/csv_prompt.json" \
    --freeze 0 \
    --learning_rate 1e-3 \
    --downsample_rate 20 \
    --output_dir [Your_Output_Path] \
```

#### (2) Explore [Time Series Prompt] and [Linear Tokenization] on gpt2-medium
```bash
python3 main_ltsm.py \
    --model LTSM \
    --model_name_or_path gpt2-medium \
    --train_epochs 500 \
    --batch_size 10 \
    --pred_len 96 \
    --data_path "datasets/ETT-small/ETTh1.csv" \
    --test_data_path_list "datasets/ETT-small/ETTh1.csv" \
    --prompt_data_path "prompt_bank/prompt_data_normalize_split" \
    --freeze 0 \
    --learning_rate 1e-3 \
    --downsample_rate 20 \
    --output_dir [Your_Output_Path] \
```

#### (3) Finetune your dataset based on pre-trained LTSM-bundle model: [Time Series Prompt] and [Linear Tokenization] on gpt2-medium
```bash
python3 main_ltsm.py \
    --model LTSM \
    --model_name_or_path gpt2-medium \
    --local_pretrain  LSC2204/LTSM-bundle \ # This model weight is for pred_len == 96
    --train_epochs 500 \
    --batch_size 10 \
    --pred_len 96 \
    --data_path "datasets/ETT-small/ETTh1.csv" \
    --test_data_path_list "datasets/ETT-small/ETTh1.csv" \
    --prompt_data_path "prompt_bank/prompt_data_normalize_split" \
    --freeze 0 \
    --learning_rate 1e-3 \
    --downsample_rate 20 \
    --output_dir [Your_Output_Path] \
```

