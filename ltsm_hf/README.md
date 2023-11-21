# Large Time Series Model (LTSM)

## Installation
```
conda create -n ltsm python=3.8.0
conda activate ltsm
git clone git@github.com:daochenzha/ltsm.git
cd ltsm
pip3 install -e .
pip3 install -r requirements.txt
```

## Quick start
Train model on Monash:
```
bash scripts/train_multigpu.sh
```

## Datasets
Path to Training Data on Monash:
```
/home/jy101/ltsm/dataset/monash/
```

Path to Prompt on Monash:
```
/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize
```
