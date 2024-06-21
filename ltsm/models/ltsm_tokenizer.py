import numpy as np
import torch
import torch.nn as nn
from torch import optim
from einops import rearrange

from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel, AutoConfig, GPT2Model, LlamaModel, GemmaModel


class LTSM_Tokenizer(PreTrainedModel):
    def __init__(self, configs):
        super().__init__(configs)
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain

        self.d_type = torch.bfloat16
        self.pred_len = configs.pred_len    

        if configs.pretrain:
            print("Loading the pretraining weight.")
            self.llm_config = AutoConfig.from_pretrained(configs.model_name_or_path)
            self.llm_model = AutoModel.from_pretrained(configs.model_name_or_path)  # loads a pretrained GPT-2 base model
        else:
            raise NotImplementedError("You must load the pretraining weight.")

        self.model_prune(configs)
        print("gpt2 = {}".format(self.llm_model))
            
    def model_prune(self, configs):

        if type(self.llm_model) == GPT2Model:
            self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
        elif type(self.llm_model) == LlamaModel or type(self.llm_model) == GemmaModel:
            self.llm_model.layers = self.llm_model.layers[:configs.gpt_layers]
        else:
            raise NotImplementedError(f"No implementation for {self.llm_model}.")

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.int()
        
        outputs = self.llm_model(input_ids = x).last_hidden_state
        outputs = outputs[:, -self.pred_len:, :]

        return outputs