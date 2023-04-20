# Large Time Series Model (LSTM)

## Installation
```
conda create -n ltsm python=3.8.0
conda activate ltsm
git clone git@github.com:daochenzha/ltsm.git
cd ltsm
pip3 install -e .
pip3 install -r requirements.txt
```

## Datasets
Let's maintain a table of the datasets. We just put datasets in our server for now without uploading to Github.

TODO: Add a table here

## Quick start
Get some example data (in datalab1 at Rice):

```
cp -r /home/dz36/ltsm/ltsm/dataset ./
```

Train model on Weather dataset:
```
python main.py --model_id test_run
```

## Roadmap

<img width="800" src="./imgs/overview.png" alt="overview" />

### Stage 1

We train the model on some datasets in the same domain to see whether it could work.

Action items:
*   Allen: Focus on modeling. Adapt the code to a form that is more suitable for pre-training and tuning hyperprameters. Develop models.
*   Guanchu: Focus on efficency. Current task: implment and test data parallel
*   Jiayi: Focus on data. Collect and process data into a format that can be directly loaded by the data loader. Do manual cleaning or filtering if needed.
*   Henry: Provide guidance and trouble shooting. Identifdy the potential good data sources, time series prerpocessing experiences, etc.
*   Daochen: Design the whole workflow and organize the efforts.

Tentative author order if we submit a paper later:
Allen*, Guanchu*, Jiayi*, Henry*, Daochen*, [some others], Xia Hu

Note:
1. \* means equal contribution
2. The order of first three authors are subject to change based on actual contribution.
3. Anyone could be removed if not contributing, as suggested by Dr. Hu.
4. [some others] are reserved for Stage 2 (no *). If Stage 1 works out, it is very likely we need more help, e.g., data.


### Stage 2
Train model with prompts. TBD


## Resources
[Power Time Series Forecasting by Pretrained LM](https://arxiv.org/pdf/2302.11939.pdf)
