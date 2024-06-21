# Understanding Different Design Choices in Training Large Time Series Models
<img width="700" height="290" src="./imgs/ltsm_model.png">

This work investigates the transition from traditional Time Series Forecasting (TSF) to Large Time Series Models (LTSMs), leveraging universal transformer-based models. Training LTSMs on diverse time series data introduces challenges due to varying frequencies, dimensions, and patterns. We explore various design choices for LTSMs, covering pre-processing, model configurations, and dataset setups. We introduce $\texttt{Time Series Prompt}$, a statistical prompting strategy, and $\texttt{LTSM-bundle}$, which encapsulates the most effective design practices identified. Our empirical results show that $\texttt{LTSM-bundle}$ outperforms existing LTSMs and traditional TSF methods in zero-shot and few-shot scenarios on benchmark datasets. $\texttt{LTSM-bundle}$  is developed by [Data Lab](https://cs.rice.edu/~xh37/) at Rice University.

## Resources
:mega: We have released our paper and training code of LTSM-bundle-v1.0!
* Paper: https://arxiv.org/abs/2406.14045
* Poster: [https://reurl.cc/5OvprR](https://arxiv.org/abs/2406.14045)
* Do you want to learn more about data pipeline search? Please check out our [data-centric AI survey](https://arxiv.org/abs/2303.10158) and [data-centric AI resources](https://github.com/daochenzha/data-centric-AI) !

## Why We Need LTSM-bundle ?


## Installation
```
conda create -n ltsm python=3.8.0
conda activate ltsm
git clone git@github.com:daochenzha/ltsm.git
cd ltsm
pip3 install -e .
pip3 install -r requirements.txt
```

## Quick Exploration on LTSM-bundle 

Training on **[Time Series Prompt]** and **[Linear Tokenization]**
```bash
bash scripts/train_ltsm_csv.sh
```

Training on **[Text Prompt]** and **[Linear Tokenization]**
```bash
bash scripts/train_ltsm_csv.sh
```

Training on **[Time Series Prompt]** and **[Time Series Tokenization]**
```bash
bash scripts/train_ltsm_tokenizer_csv.sh
```

## Datasets and time series prompts
Please download the datasets and prompts before training
```bash
cd datasets
download: 
```

## Cite This Work
If you find this work useful, you may cite this work:
```

```